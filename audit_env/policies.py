"""Четыре базовые политики организаторов. Интерфейс: policy.act(observation) -> action.

* RandomExplorer   — случайный исследователь с фиксированным seed.
* SimpleReconciler — простая сверка: читает все таблицы и ищет расхождения упрощённым детектором.
* FlagAll          — атака «пометить всё».
* Oracle           — только для оценщика: знает эталон, шлёт верные выводы с доказательствами (потолок награды).
"""
import random

from .env import MAX_STEPS, TABLES
from .grader.validate import detect

FINDING_TYPES = ("duplicate_invoice", "payment_allocation", "fx_error", "period_cutoff")


def _submit(episode_id, findings):
    return {"tool": "submit", "args": {"findings": {"episode_id": episode_id, "findings": findings}}}


class Policy:
    needs_truth = False
    name = "policy"

    def start(self, observation, info):
        self.episode_id = info["episode_id"]
        self.first = observation

    def act(self, observation):
        raise NotImplementedError


class RandomExplorer(Policy):
    name = "random_explorer"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def start(self, observation, info):
        super().start(observation, info)
        self.todo = self.rng.randint(5, 15)
        self.seen = {"invoices": [], "payments": []}
        self.amounts = []

    def _remember(self, observation):
        result = observation.get("result") or {}
        for row in result.get("records", []):
            if "invoice_id" in row and "payment_id" not in row:
                self.seen["invoices"].append(row["invoice_id"])
                self.amounts.append(row["total_minor"])
            elif "payment_id" in row and "doc_id" in row:
                self.seen["payments"].append(row["payment_id"])
                self.amounts.append(abs(row["amount_minor"]))

    def act(self, observation):
        self._remember(observation)
        if self.todo > 0:
            self.todo -= 1
            table = self.rng.choice(TABLES)
            total = self.first["tables"][table]
            cursor = None if self.rng.random() < 0.4 else str(self.rng.randrange(0, max(total, 1)))
            return {"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": self.rng.randint(5, 50)}}
        findings = []
        for _ in range(self.rng.randint(0, 2)):
            kind = self.rng.choice(FINDING_TYPES)
            pool = self.seen["payments"] if kind in ("payment_allocation", "fx_error") else self.seen["invoices"]
            if not pool or not self.amounts:
                continue
            ids = [self.rng.choice(pool)]
            if kind == "duplicate_invoice":
                ids.append(self.rng.choice(pool))
            findings.append({"type": kind, "affected_ids": ids, "claimed_amount_minor": self.rng.choice(self.amounts),
                             "currency": self.first["manifest"]["base_currency"], "evidence_ids": [],
                             "explanation": "random guess"})
        return _submit(self.episode_id, findings)


class SimpleReconciler(Policy):
    """Читает таблицы постранично и запускает упрощённый детектор (naive=True):
    курс всегда по дате UTC, составы line_items не сверяются (их нет в списках)."""
    name = "simple_reconciler"

    def start(self, observation, info):
        super().start(observation, info)
        self.queue = [("invoices", None), ("payments", None), ("fx", None), ("ledger", None)]
        self.data = {"manifest": observation["manifest"], "policy": observation["policy"],
                     "ledger": [], "invoices": [], "payments": [], "fx": []}
        self.current = None

    def act(self, observation):
        result = observation.get("result") or {}
        if self.current and result.get("records") is not None:
            self.data[self.current].extend(result["records"])
            if result["next_cursor"] is not None:
                self.queue.insert(0, (self.current, result["next_cursor"]))
        if self.queue and observation["budget_remaining"] > 1:
            self.current, cursor = self.queue.pop(0)
            return {"tool": "list_records", "args": {"table": self.current, "cursor": cursor, "limit": 50}}
        try:
            found = detect(self.data, naive=True) if all(self.data[t] for t in ("invoices", "payments", "ledger", "fx")) else []
        except KeyError:  # шагов не хватило прочитать все таблицы: данных для сверки мало
            found = []
        return _submit(self.episode_id, found)


class FlagAll(Policy):
    """Атака: помечаем каждый увиденный счёт и платёж каждым подходящим типом."""
    name = "flag_all"

    def start(self, observation, info):
        super().start(observation, info)
        self.step_no = 0
        self.invoices, self.payments = [], []

    def act(self, observation):
        result = observation.get("result") or {}
        for row in result.get("records", []):
            (self.invoices if "invoice_id" in row and "payment_id" not in row else self.payments).append(row)
        self.step_no += 1
        if self.step_no == 1:
            return {"tool": "list_records", "args": {"table": "invoices", "limit": 50}}
        if self.step_no == 2:
            return {"tool": "list_records", "args": {"table": "payments", "limit": 50}}
        findings = []
        for inv in self.invoices:
            findings.append({"type": "period_cutoff", "affected_ids": [inv["invoice_id"]], "claimed_amount_minor": inv["total_minor"],
                             "currency": inv["currency"], "evidence_ids": [inv["doc_id"]], "explanation": "flag all"})
        for pay in self.payments:
            for kind in ("fx_error", "payment_allocation"):
                findings.append({"type": kind, "affected_ids": [pay["payment_id"]], "claimed_amount_minor": abs(pay["amount_minor"]),
                                 "currency": pay["currency"], "evidence_ids": [pay["doc_id"]], "explanation": "flag all"})
        findings.sort(key=lambda f: (f["type"], f["affected_ids"]))
        step = max(1, len(findings) // 30)
        return _submit(self.episode_id, findings[::step][:30])


class Oracle(Policy):
    """Знает закрытый эталон. Проверяет максимум награды, а не способность агента."""
    name = "oracle"
    needs_truth = True

    def __init__(self, truth=None):
        self.truth = truth

    def act(self, observation):
        findings = [{"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
                     "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "oracle"}
                    for i in self.truth["issues"]]
        return _submit(self.episode_id, findings)


def make_policy(name: str, seed: int = 0, truth=None) -> Policy:
    if name == "random_explorer":
        return RandomExplorer(seed)
    if name == "simple_reconciler":
        return SimpleReconciler()
    if name == "flag_all":
        return FlagAll()
    if name == "oracle":
        return Oracle(truth)
    raise ValueError(f"unknown policy: {name}")


POLICY_NAMES = ("random_explorer", "simple_reconciler", "flag_all", "oracle")
assert MAX_STEPS == 100
