"""Проверка структуры findings.json и общий matcher по опубликованным «Определениям проверок FinTech».

Нормализация (точные соглашения жюри):
* ID сравниваются точно: с учётом регистра и пробелов, без обрезки и поиска похожих.
* affected_ids — множество точных ID: порядок и повторы внутри списка не важны.
  Для TP множество должно точно совпасть с каноническим набором эталонной ошибки.
* evidence_ids — множество после проверки исходного лимита (10 элементов вместе с повторами).
  TP требует полного включения хотя бы одного допустимого набора. Лишние СУЩЕСТВУЮЩИЕ доказательства
  допустимы; несуществующий или недоступный ID делает вывод несопоставимым (FP).
* type и currency — точное совпадение строк, без смены регистра и обрезки.
* claimed_amount_minor — JSON-целое (не bool, не строка, не дробное); сравнение в минимальных единицах
  с допуском τ эталона. τ = 0 для всех типов этой среды (суммы считаются точно, см. TOLERANCE).

Канонические affected_ids по типам (публикуем заранее, в коде генератора и здесь):
* duplicate_invoice  — {invoice_id оригинала, invoice_id дубля}
* payment_allocation — {payment_id}
* fx_error           — {payment_id}
* period_cutoff      — {invoice_id}

Структура против несовпадения:
* Неверный тип значения, отсутствующее обязательное поле, превышение лимита (31 вывод, 11 ID
  доказательств в выводе, explanation длиннее 1000 символов Unicode) — неверная структура всего submit,
  награда -1. Лишние ключи игнорируются.
* Структурно верный вывод с неправильным type, affected_ids, суммой, валютой или доказательствами — FP.

Сопоставление: двудольный граф (выводы × ошибки эталона), ребро только при всех условиях TP,
паросочетание максимальной мощности. Среди максимальных выбирается детерминированно: ошибки эталона
по issue_id в порядке кодовых точек Unicode, для каждой — наименьший индекс вывода, при котором
максимальная мощность ещё достижима. Порядок findings не меняет TP, FP, FN и награду.
"""
from dataclasses import dataclass, field

from .reward import f1_score, precision, recall

MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_EXPLANATION = 1000
# Защита оценщика от огромного ввода (в кейсе не заданы; заведомо больше любого честного ответа)
MAX_AFFECTED = 100
MAX_STRING = 256
REQUIRED_KEYS = ("type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation")


class StructureError(ValueError):
    """Неверная структура findings.json: эпизод завершается с наградой -1."""


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_issue_ids: list = field(default_factory=list)
    pairs: list = field(default_factory=list)  # [(issue_id, индекс вывода)]

    @property
    def precision(self):
        return precision(self.tp, self.fp)

    @property
    def recall(self):
        return recall(self.tp, self.fn)

    @property
    def f1(self):
        return f1_score(self.tp, self.fp, self.fn)


def _string(value, name):
    if not isinstance(value, str):
        raise StructureError(f"{name} must be a string")
    if len(value) > MAX_STRING:
        raise StructureError(f"{name} is longer than {MAX_STRING} characters")
    return value


def _string_list(value, name, limit):
    if not isinstance(value, list):
        raise StructureError(f"{name} must be a list of strings")
    if len(value) > limit:  # исходный размер, до удаления повторов (и до цикла: быстро на огромном вводе)
        raise StructureError(f"{name}: at most {limit} items")
    return [_string(item, f"{name} item") for item in value]


def validate_submission(payload, episode_id: str) -> list:
    """Проверяет findings.json и возвращает выводы с множествами ID. Бросает StructureError."""
    if not isinstance(payload, dict) or "episode_id" not in payload or "findings" not in payload:
        raise StructureError("findings.json must be an object with episode_id and findings")
    if payload["episode_id"] != episode_id:
        raise StructureError("episode_id mismatch")
    findings = payload["findings"]
    if not isinstance(findings, list):
        raise StructureError("findings must be a list")
    if len(findings) > MAX_FINDINGS:
        raise StructureError(f"at most {MAX_FINDINGS} findings")
    result = []
    for raw in findings:
        if not isinstance(raw, dict):
            raise StructureError("each finding must be an object")
        missing = [key for key in REQUIRED_KEYS if key not in raw]
        if missing:
            raise StructureError("finding is missing: " + ", ".join(missing))
        amount = raw["claimed_amount_minor"]
        if isinstance(amount, bool) or not isinstance(amount, int):
            raise StructureError("claimed_amount_minor must be a JSON integer")
        explanation = raw["explanation"]
        if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION:
            raise StructureError(f"explanation must be a string of at most {MAX_EXPLANATION} characters")
        result.append({
            "type": _string(raw["type"], "type"),
            "affected_ids": frozenset(_string_list(raw["affected_ids"], "affected_ids", MAX_AFFECTED)),
            "claimed_amount_minor": amount,
            "currency": _string(raw["currency"], "currency"),
            "evidence_ids": frozenset(_string_list(raw["evidence_ids"], "evidence_ids", MAX_EVIDENCE)),
            "explanation": explanation,
        })
    return result


def _is_edge(finding: dict, issue: dict, universe) -> bool:
    """Все условия TP для пары (вывод, ошибка эталона)."""
    if finding["type"] != issue["type"] or finding["currency"] != issue["currency"]:
        return False
    if finding["affected_ids"] != frozenset(issue["affected_ids"]):
        return False
    if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > issue.get("tolerance_minor", 0):
        return False
    evidence = finding["evidence_ids"]
    if universe is not None and not evidence <= universe:
        return False  # несуществующее или недоступное доказательство
    return any(frozenset(s) <= evidence for s in issue["evidence_sets"])


def _max_matching(adj: dict, issues: list, used: frozenset) -> int:
    """Мощность максимального паросочетания (алгоритм Куна) для issues без уже занятых выводов."""
    owner = {}

    def augment(issue, seen):
        for f in adj[issue]:
            if f in used or f in seen:
                continue
            seen.add(f)
            if f not in owner or augment(owner[f], seen):
                owner[f] = issue
                return True
        return False

    return sum(1 for issue in issues if augment(issue, set()))


def match_findings(findings: list, issues: list, universe=None) -> MatchResult:
    """findings — результат validate_submission; issues — список из truth.json;
    universe — множество всех существующих ID записей эпизода (None — не проверять существование)."""
    universe = frozenset(universe) if universe is not None else None
    order = sorted(issues, key=lambda i: i["issue_id"])
    ids = [i["issue_id"] for i in order]
    adj = {i["issue_id"]: [n for n, f in enumerate(findings) if _is_edge(f, i, universe)] for i in order}
    best = _max_matching(adj, ids, frozenset())
    pairs, used = [], frozenset()
    for k, issue_id in enumerate(ids):
        for n in adj[issue_id]:
            if n in used:
                continue
            if len(pairs) + 1 + _max_matching(adj, ids[k + 1:], used | {n}) == best:
                pairs.append((issue_id, n))
                used = used | {n}
                break
    tp = len(pairs)
    return MatchResult(tp=tp, fp=len(findings) - tp, fn=len(issues) - tp,
                       matched_issue_ids=[p[0] for p in pairs], pairs=pairs)


def evidence_universe(data: dict) -> frozenset:
    """Все ID, по которым агент может получить запись инструментами: row_id, invoice_id, payment_id,
    doc_id и id курса 'ДАТА:SRC:DST'."""
    ids = {r["row_id"] for r in data["ledger"]}
    ids |= {i["invoice_id"] for i in data["invoices"]} | {i["doc_id"] for i in data["invoices"]}
    ids |= {p["payment_id"] for p in data["payments"]} | {p["doc_id"] for p in data["payments"]}
    ids |= {f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in data["fx"]}
    return frozenset(ids)
