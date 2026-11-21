"""AuditEnv: среда в стиле Gymnasium (JSON-совместимый контракт, наследовать Env не нужно).

reset(seed, config) -> (observation, info)
step(action)        -> (observation, reward, terminated, truncated, info)

Действие: {"tool": <имя>, "args": {...}}. Инструменты: list_records, get_record, get_document,
calculator, submit. Свободного shell, SQL, файловой системы и интернета нет.

Режимы:
* "open"   (обучение): терминальная награда возвращается политике.
* "closed" (проверка): награда и диагностика остаются у оценщика, политика получает 0.0.
              Итог читает только проверяющий: env.final_report().

Важно: final_report() и evaluator_truth() предназначены для проверяющего. Если агент работает
в том же процессе и держит объект env, он до них дотянется. Настоящая изоляция даёт только запуск
среды отдельным процессом (audit_env.server): агент видит лишь ответы на команды.

Лимит времени (60 с) — по определениям проверок жюри: от начала reset до завершения эпизода,
включая работу политики между вызовами. Шаг, на котором лимит превышен (в том числе submit),
усекает эпизод: truncated=true, итоговый ответ пустой.
"""
import copy
import time

from pathlib import Path

from . import calculator, files, generator
from .grader.evaluator import Evaluator
from .grader.matcher import StructureError, evidence_universe, validate_submission

MAX_STEPS = 100
TIME_LIMIT_S = 60.0
PAGE_MAX = 50
TABLES = ("ledger", "invoices", "payments", "fx")
TOOLS = ("list_records", "get_record", "get_document", "calculator", "submit")

TASK = """You are an AI auditor. One company, one reporting period. The books may contain hidden errors
of four types (or none). Investigate with the tools, then call submit once with your findings.

Tools (each call costs one step; max 100 steps including submit; unknown ids/tools also cost a step):
  list_records {table, cursor?, limit?}  tables: ledger, invoices, payments, fx; page <= 50 rows
  get_record   {table, id}               id = row_id / invoice_id / payment_id / fx id "DATE:SRC:DST"
  get_document {doc_id}                  full invoice (with line_items) or payment document
  calculator   {expression}              + - * / ( ) and decimal numbers only
  submit       {findings: {episode_id, findings: [...]}}   ends the episode

finding = {type, affected_ids, claimed_amount_minor, currency, evidence_ids, explanation}
  max 30 findings, 10 evidence ids, 1000 characters of explanation; no other keys.
  Amounts are integers in minor units and must match exactly (tolerance 0). Repeats count as false positives.
  IDs, type and currency are compared exactly (case and spaces matter). affected_ids is a set.
  evidence_ids: existing doc_id, invoice_id, payment_id, ledger row_id or fx id that prove the finding. A finding is
  correct only with a sufficient set; extra existing ids are fine, an id that does not exist makes it wrong.

Types:
  duplicate_invoice  affected_ids = [original invoice_id, duplicate invoice_id]; amount = invoice total; currency = invoice
                     currency; evidence = doc_ids of both invoices.
  payment_allocation affected_ids = [payment_id]; amount = part of the payment parked in UNAPPLIED instead of AR (invoice
                     currency); evidence = payment doc_id + that UNAPPLIED row_id.
  fx_error           affected_ids = [payment_id]; amount = |booked - correct| for the payment legs in invoice currency;
                     evidence = payment doc_id + its AR/UNAPPLIED row_ids + the correct fx id.
  period_cutoff      affected_ids = [invoice_id]; amount = invoice total; evidence = invoice doc_id + its AR or REVENUE row_id.
A clean company has no errors: then submit an empty findings list. False accusations are punished.
Reward = clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1).
"""


def _has_truth(folder) -> bool:
    folder = Path(folder)
    return (folder / files.CLOSED_DIR / "truth.json").is_file() or (folder / "truth.json").is_file()


class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _short(value, limit=40):
    return str(value)[:limit]


JUDGE_KEYS = frozenset({"mode", "episode_dir", "team_config"})


class AuditEnv:
    def __init__(self, mode: str = "open", clock=time.monotonic):
        if mode not in ("open", "closed"):
            raise ValueError("mode must be 'open' or 'closed'")
        self.mode = mode
        self._clock = clock
        self._state = None

    # ------------------------------------------------------------ жизненный цикл
    def _load(self, seed, config):
        """Откуда взять эпизод и в каком режиме играть.

        Конфиг жюри: {"mode": "closed"|"train", "episode_dir": "<папка>", "team_config": {...}}.
          * episode_dir задан — эпизод загружается из папки как есть (ID и порядок не меняются).
            В закрытом режиме эталон не читается вовсе: оценку делает score_episode у жюри.
          * episode_dir нет — эпизод генерируется по seed и team_config.
        Любой другой config — это config генератора (режим берётся из AuditEnv(mode=...)).
        """
        if isinstance(config, dict) and config and set(config) <= JUDGE_KEYS:
            name = config.get("mode", "train")
            if name not in ("train", "open", "closed"):
                raise ValueError("mode must be 'closed' or 'train'")
            mode = "closed" if name == "closed" else "open"
            if config.get("episode_dir"):
                folder = config["episode_dir"]
                closed_needed = mode == "open" and _has_truth(folder)
                return mode, files.load_episode(folder, with_closed=closed_needed)
            return mode, generator.generate(seed, config.get("team_config"))
        return self.mode, generator.generate(seed, config)

    def reset(self, seed: int = 0, config: dict | None = None):
        started = self._clock()  # лимит времени эпизода считается от начала reset
        mode, episode = self._load(seed, config)
        tables = {"ledger": episode.ledger,
                  "invoices": [{k: v for k, v in i.items() if k != "line_items"} for i in episode.invoices],
                  "payments": episode.payments,
                  "fx": episode.fx}
        index = {
            "ledger": {r["row_id"]: r for r in episode.ledger},
            "invoices": {i["invoice_id"]: i for i in episode.invoices},
            "payments": {p["payment_id"]: p for p in episode.payments},
            "fx": {f"{r['date']}:{r['source_currency']}:{r['target_currency']}": r for r in episode.fx},
        }
        docs = {i["doc_id"]: ("invoice", i) for i in episode.invoices}
        docs.update({p["doc_id"]: ("payment", p) for p in episode.payments})
        evaluator = None
        if episode.truth is not None:
            evaluator = Evaluator(episode.truth, evidence_universe(
                {"ledger": episode.ledger, "invoices": episode.invoices, "payments": episode.payments, "fx": episode.fx}))
        self._state = {"episode": episode, "mode": mode, "tables": tables, "index": index, "docs": docs, "step": 0,
                       "done": False, "started": started, "evaluator": evaluator, "report": None, "submission": None}
        obs = {"task": TASK, "manifest": copy.deepcopy(episode.manifest), "policy": copy.deepcopy(episode.policy),
               "tables": {name: len(rows) for name, rows in tables.items()},
               "step": 0, "budget_remaining": MAX_STEPS, "result": None, "error": None}
        return obs, {"episode_id": episode.episode_id}

    def _elapsed(self):
        return self._clock() - self._state["started"]

    def _grade(self, payload, inspection_steps):
        """Оценка ответа. Без эталона (закрытый режим из папки) — только проверка структуры."""
        state = self._state
        if state["evaluator"] is not None:
            return state["evaluator"].grade(payload, inspection_steps)
        try:
            validate_submission(payload, state["episode"].episode_id)
            status, problem = "submitted", None
        except StructureError as exc:
            status, problem = "invalid", str(exc)
        return {"status": status, "error": problem, "scored": False, "reward": -1.0 if status == "invalid" else 0.0,
                "inspection_steps": inspection_steps}

    def step(self, action):
        state = self._state
        if state is None:
            raise RuntimeError("call reset() first")
        if state["done"]:
            raise RuntimeError("episode is over; call reset()")
        state["step"] += 1
        step = state["step"]
        result = error = None
        terminated = truncated = False
        report = None

        if self._elapsed() > TIME_LIMIT_S:
            truncated, error = True, {"code": "time_limit", "message": "time limit reached"}
        else:
            try:
                tool, args = self._parse(action)
                if tool == "submit":
                    terminated = True
                    state["submission"] = self._submission(args)
                    report = self._grade(state["submission"], step - 1)
                    if report["status"] == "invalid":
                        error = {"code": "invalid_submission", "message": "submission rejected"}
                    else:
                        result = {"status": "submitted"}
                else:
                    result = self._run(tool, args)
            except ToolError as exc:
                error = {"code": exc.code, "message": exc.message}
            if not terminated and step >= MAX_STEPS:
                truncated = True  # лимит шагов: итоговый ответ пустой
            if self._elapsed() > TIME_LIMIT_S:
                # лимит превышен на этом шаге (даже если это submit): эпизод усечён, ответ пустой
                terminated, truncated = False, True
                result, error = None, {"code": "time_limit", "message": "time limit reached"}
        if truncated:
            state["submission"] = None
            report = self._grade({"episode_id": state["episode"].episode_id, "findings": []}, step)
        done = terminated or truncated
        if done:
            state["done"] = True
            report.update({"episode_id": state["episode"].episode_id, "steps": step, "terminated": terminated,
                           "truncated": truncated, "elapsed_s": round(self._elapsed(), 3)})
            state["report"] = report
        reward = report["reward"] if done and state["mode"] == "open" else 0.0
        obs = {"step": step, "budget_remaining": max(MAX_STEPS - step, 0), "result": result, "error": error}
        return obs, float(reward), terminated, truncated, {"episode_id": state["episode"].episode_id, "step": step}

    # ------------------------------------------------------------ только для проверяющего
    def final_report(self) -> dict:
        """Итог эпизода (метрики, награда, журнал). Доступен после завершения и не нужен политике."""
        if self._state is None or not self._state["done"]:
            raise RuntimeError("the episode is not finished")
        report = dict(self._state["report"])
        evaluator = self._state["evaluator"]
        report["evaluator_log"] = list(evaluator.log) if evaluator else []
        report["submission"] = copy.deepcopy(self._state["submission"])
        return report

    def evaluator_truth(self) -> dict:
        """Закрытый эталон для oracle и тестов. Среда не включает его ни в одно наблюдение."""
        if self._state is None:
            raise RuntimeError("call reset() first")
        if self._state["evaluator"] is None:
            raise RuntimeError("this episode was loaded without the evaluator files")
        return self._state["evaluator"].truth()

    # ------------------------------------------------------------ инструменты
    @staticmethod
    def _parse(action):
        if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
            raise ToolError("invalid_action", "action must be {tool, args}")
        args = action.get("args", {})
        if not isinstance(args, dict):
            raise ToolError("invalid_action", "args must be an object")
        if action["tool"] not in TOOLS:
            raise ToolError("unknown_tool", "unknown tool")
        return action["tool"], args

    @staticmethod
    def _submission(args):
        # findings=None или лишние ключи — неверная структура (None не равен пустому ответу)
        return args["findings"] if set(args) == {"findings"} else {"invalid": True}

    def _table(self, args):
        table = args.get("table")
        if table not in TABLES:
            raise ToolError("invalid_args", "table must be one of " + ", ".join(TABLES))
        return table

    def _run(self, tool, args):
        state = self._state
        if tool == "list_records":
            table = self._table(args)
            rows = state["tables"][table]
            limit = args.get("limit", PAGE_MAX)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ToolError("invalid_args", "limit must be a positive integer")
            limit = min(limit, PAGE_MAX)
            cursor = args.get("cursor")
            if cursor is None:
                offset = 0
            elif (isinstance(cursor, str) and cursor.isascii() and cursor.isdigit() and len(cursor) <= 9
                  and int(cursor) <= len(rows)):
                offset = int(cursor)
            else:
                raise ToolError("invalid_cursor", "invalid cursor")
            nxt = offset + limit
            return {"table": table, "records": copy.deepcopy(rows[offset:nxt]), "total": len(rows),
                    "next_cursor": str(nxt) if nxt < len(rows) else None}
        if tool == "get_record":
            table = self._table(args)
            record = state["index"][table].get(args.get("id")) if isinstance(args.get("id"), str) else None
            if record is None:
                raise ToolError("not_found", f"no such id: {_short(args.get('id'))}")
            return {"table": table, "record": copy.deepcopy(record)}
        if tool == "get_document":
            found = state["docs"].get(args.get("doc_id")) if isinstance(args.get("doc_id"), str) else None
            if found is None:
                raise ToolError("not_found", f"no such document: {_short(args.get('doc_id'))}")
            return {"doc_type": found[0], "document": copy.deepcopy(found[1])}
        if tool == "calculator":
            try:
                return {"value": calculator.evaluate(args.get("expression"))}
            except calculator.CalcError as exc:
                raise ToolError("calc_error", str(exc))
        raise ToolError("unknown_tool", "unknown tool")  # недостижимо: submit обрабатывается в step
