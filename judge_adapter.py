"""Адаптер жюри по «Определениям проверок FinTech» (раздел «Единый способ подключения»).

Точки входа:
    generate_episode(seed, config, output_dir)        — эпизод в папку жюри (открытые + закрытые файлы)
    make_env()                                         — объект с reset(seed, config) и step(action)
    score_episode(episode_dir, findings, inspection_steps) — закрытая оценка; агенту не доступна

Рабочая директория — корень репозитория (в контейнере /app). Импорт: `import judge_adapter`.

Формат обмена (JSON-совместимые значения):
    reset(seed, {"mode": "closed", "episode_dir": "<папка>", "team_config": {...}})
        -> ({"task", "manifest", "policy", "remaining_steps": 100}, {"episode_id"})
    step({"name": "<действие>", "args": {...}})
        -> ({"result", "error", "remaining_steps"}, reward, terminated, truncated, {"episode_id", "step"})
    Действия и аргументы: list_records{table, cursor, limit}, get_record{table, id}, get_document{doc_id},
    calculator{expression}, submit{findings}. Таблицы: ledger, invoices, payments, fx.
    Результаты: list_records -> {"records": [...], "next_cursor": str|null}; get_record и get_document ->
    объект записи или документа; calculator -> десятичная строка; submit -> {"accepted": true}.
    Ошибка: {"code": "...", "message": "..."} без путей, seed и закрытых данных; шаг расходуется.
    В закрытом режиме reward всегда 0 (и на завершении); в train терминальная награда доступна политике.

Типы ошибок (поле type): duplicate_invoice, payment_allocation, fx_error, period_cutoff.
Канонические affected_ids и правила сопоставления — в audit_env/grader/matcher.py.
"""
import copy
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_env.env import MAX_STEPS, AuditEnv  # noqa: E402
from audit_env.files import load_open, load_truth, write_episode  # noqa: E402
from audit_env.generator import generate  # noqa: E402
from audit_env.grader.evaluator import Evaluator  # noqa: E402
from audit_env.grader.matcher import evidence_universe  # noqa: E402

log = logging.getLogger("audit_env.evaluator")


def _team_config(config):
    """Конфиг генератора: сам config или его поле team_config (оба варианта допустимы)."""
    if isinstance(config, dict) and "team_config" in config:
        return config["team_config"]
    return config or None


def generate_episode(seed: int, config: dict, output_dir: str) -> None:
    """Создаёт эпизод в output_dir: manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl,
    fx.csv и закрытые evaluator/truth.json, evaluator/provenance.json. Обёртка над генератором команды."""
    write_episode(generate(seed, _team_config(config)), output_dir, with_closed=True)


class JudgeEnv:
    """Среда в формате жюри поверх AuditEnv. Переводит оболочку действий и форму ответов,
    не выбирая действия за политику и не меняя смысл задачи."""

    def __init__(self, clock=time.monotonic):
        self._env = AuditEnv(clock=clock)
        self._last = None  # последнее наблюдение и флаги завершения

    def reset(self, seed: int = 0, config: dict | None = None):
        obs, info = self._env.reset(seed, config)
        self._last = {"remaining_steps": MAX_STEPS, "terminated": False, "truncated": False,
                      "episode_id": info["episode_id"], "step": 0}
        first = {"task": obs["task"], "manifest": obs["manifest"], "policy": obs["policy"],
                 "remaining_steps": obs["budget_remaining"]}
        return first, dict(info)

    def step(self, action):
        if self._last is None:
            return ({"result": None, "error": {"code": "not_started", "message": "call reset first"},
                     "remaining_steps": 0}, 0.0, False, False, {})
        if self._last["terminated"] or self._last["truncated"]:
            # эпизод завершён: ничего не меняется, итог уже зафиксирован
            return ({"result": None, "error": {"code": "episode_over", "message": "episode is over; call reset"},
                     "remaining_steps": self._last["remaining_steps"]}, 0.0, self._last["terminated"],
                    self._last["truncated"], {"episode_id": self._last["episode_id"], "step": self._last["step"]})
        native = _to_native(action)
        obs, reward, terminated, truncated, info = self._env.step(native)
        tool = native.get("tool") if isinstance(native, dict) else None
        result = _to_judge_result(tool, obs["result"]) if obs["error"] is None else None
        self._last = {"remaining_steps": obs["budget_remaining"], "terminated": terminated, "truncated": truncated,
                      "episode_id": info["episode_id"], "step": info["step"]}
        return ({"result": result, "error": obs["error"], "remaining_steps": obs["budget_remaining"]},
                reward, terminated, truncated, info)


def _to_native(action):
    """{"name", "args"} -> {"tool", "args"}. Наш собственный формат {"tool", ...} тоже принимается."""
    if isinstance(action, dict) and "name" in action and "tool" not in action:
        return {"tool": action["name"], "args": action.get("args", {})}
    return action


def _to_judge_result(tool, result):
    if result is None:
        return None
    if tool == "list_records":
        return {"records": result["records"], "next_cursor": result["next_cursor"]}
    if tool == "get_record":
        return result["record"]
    if tool == "get_document":
        return result["document"]
    if tool == "calculator":
        return result["value"]
    if tool == "submit":
        return {"accepted": True}
    return copy.deepcopy(result)


def make_env() -> JudgeEnv:
    return JudgeEnv()


def score_episode(episode_dir: str, findings: dict, inspection_steps: int) -> dict:
    """Закрытая оценка ответа по эталону эпизода. Валидный submit -> метрики и return;
    невалидный -> return=-1, остальные поля null, ошибка в журнале оценщика (logger audit_env.evaluator)."""
    data = load_open(episode_dir)
    evaluator = Evaluator(load_truth(episode_dir), evidence_universe(data))
    report = evaluator.grade(findings, inspection_steps)
    for line in evaluator.log:
        log.warning("%s: %s", data["manifest"].get("episode_id"), line)
    keys = ("tp", "fp", "fn", "precision", "recall", "f1")
    if report["status"] == "invalid":
        return {**{key: None for key in keys}, "return": -1.0}
    return {**{key: report[key] for key in keys}, "return": report["reward"]}


def run_policy(policy_name: str, episode_dir: str, policy_seed: int = 0) -> dict:
    """Одна из четырёх базовых политик через этот адаптер в закрытом режиме (как у жюри) и оценка
    score_episode. Oracle получает эталон только здесь, на стороне проверяющего."""
    from audit_env.policies import make_policy  # noqa: E402  (импорт здесь: политикам адаптер не нужен)

    env = make_env()
    obs, info = env.reset(0, {"mode": "closed", "episode_dir": episode_dir, "team_config": {}})
    truth = load_truth(episode_dir) if policy_name == "oracle" else None
    policy = make_policy(policy_name, policy_seed, truth)
    policy.start(obs, info)
    steps, findings, done = 0, None, False
    while not done:
        action = policy.act(obs)
        native = _to_native(action)
        if isinstance(native, dict) and native.get("tool") == "submit":
            findings = native.get("args", {}).get("findings")
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        done = terminated or truncated
        if truncated:
            findings = None  # усечение: итоговый ответ пустой
    inspection = steps - 1 if findings is not None else steps
    if findings is None:
        findings = {"episode_id": info["episode_id"], "findings": []}
    return score_episode(episode_dir, findings, inspection)
