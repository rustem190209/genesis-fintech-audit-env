"""Проверка структуры выводов и общий matcher (взаимно однозначное сопоставление с эталоном).

Правила сопоставления (открыты по требованию кейса):
* Вывод и эталонная ошибка совпадают по ключу: тот же type и тот же набор affected_ids (порядок не важен).
* TP требует ещё: ту же валюту; сумму в допуске tolerance_minor эталона; хотя бы один
  полный допустимый набор доказательств внутри evidence_ids вывода.
* Один эталон закрывается одним выводом. При нескольких кандидатах TP получает первый по порядку
  в findings, который проходит ВСЕ проверки. Остальные (повторы, неточные) считаются FP.
* Нормализация: type в нижнем регистре, пробелы по краям обрезаются, ID без дублей, валюта в верхнем регистре.
"""
from dataclasses import dataclass, field

from .reward import f1_score, precision, recall

MAX_FINDINGS = 30
MAX_EVIDENCE = 10
MAX_AFFECTED = 10
MAX_EXPLANATION = 1000
MAX_ID_LEN, MAX_TYPE_LEN, MAX_CURRENCY_LEN = 64, 32, 8
FINDING_KEYS = {"type", "affected_ids", "claimed_amount_minor", "currency", "evidence_ids", "explanation"}


class StructureError(ValueError):
    """Неверная структура findings.json: эпизод завершается с наградой -1."""


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    matched_issue_ids: list = field(default_factory=list)

    @property
    def precision(self):
        return precision(self.tp, self.fp)

    @property
    def recall(self):
        return recall(self.tp, self.fn)

    @property
    def f1(self):
        return f1_score(self.tp, self.fp, self.fn)


def _id_list(value, name, limit, allow_empty):
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise StructureError(f"{name} must be a list of strings")
    cleaned = []
    for item in value:
        if len(item) > MAX_ID_LEN:
            raise StructureError(f"{name}: id longer than {MAX_ID_LEN} characters")
        item = item.strip()
        if item and item not in cleaned:
            cleaned.append(item)
    if len(value) > limit:
        raise StructureError(f"{name}: at most {limit} items")
    if not cleaned and not allow_empty:
        raise StructureError(f"{name} must not be empty")
    return cleaned


def validate_submission(payload, episode_id: str) -> list:
    """Проверяет findings.json и возвращает нормализованные выводы. Бросает StructureError."""
    if not isinstance(payload, dict) or set(payload) != {"episode_id", "findings"}:
        raise StructureError("findings.json must be an object with keys episode_id and findings")
    if payload["episode_id"] != episode_id:
        raise StructureError("episode_id mismatch")
    findings = payload["findings"]
    if not isinstance(findings, list) or len(findings) > MAX_FINDINGS:
        raise StructureError(f"findings must be a list of at most {MAX_FINDINGS} items")
    result = []
    for raw in findings:
        if not isinstance(raw, dict) or set(raw) != FINDING_KEYS:
            raise StructureError("each finding needs exactly: " + ", ".join(sorted(FINDING_KEYS)))
        amount = raw["claimed_amount_minor"]
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise StructureError("claimed_amount_minor must be a non-negative integer")
        if not isinstance(raw["type"], str) or not isinstance(raw["currency"], str):
            raise StructureError("type and currency must be strings")
        if len(raw["type"]) > MAX_TYPE_LEN or len(raw["currency"]) > MAX_CURRENCY_LEN:
            raise StructureError("type or currency is too long")
        if not isinstance(raw["explanation"], str) or len(raw["explanation"]) > MAX_EXPLANATION:
            raise StructureError(f"explanation must be a string of at most {MAX_EXPLANATION} characters")
        result.append({
            "type": raw["type"].strip().lower(),
            "affected_ids": sorted(_id_list(raw["affected_ids"], "affected_ids", MAX_AFFECTED, False)),
            "claimed_amount_minor": amount,
            "currency": raw["currency"].strip().upper(),
            "evidence_ids": _id_list(raw["evidence_ids"], "evidence_ids", MAX_EVIDENCE, True),
            "explanation": raw["explanation"],
        })
    return result


def _fully_matches(finding: dict, issue: dict) -> bool:
    if finding["currency"] != issue["currency"]:
        return False
    if abs(finding["claimed_amount_minor"] - issue["expected_amount_minor"]) > issue["tolerance_minor"]:
        return False
    provided = set(finding["evidence_ids"])
    return any(set(s) <= provided for s in issue["evidence_sets"])


def match_findings(findings: list, issues: list) -> MatchResult:
    """findings уже нормализованы validate_submission; issues — список из truth.json."""
    by_key = {(i["type"], tuple(sorted(i["affected_ids"]))): i for i in issues}
    taken, matched = set(), []
    for finding in findings:
        issue = by_key.get((finding["type"], tuple(finding["affected_ids"])))
        if issue and issue["issue_id"] not in taken and _fully_matches(finding, issue):
            taken.add(issue["issue_id"])
            matched.append(issue["issue_id"])
    tp = len(matched)
    return MatchResult(tp=tp, fp=len(findings) - tp, fn=len(issues) - tp, matched_issue_ids=matched)
