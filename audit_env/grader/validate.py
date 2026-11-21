"""Проверка данных эпизода и независимый детектор.

Детектор смотрит только на открытые файлы (как аудитор с полным доступом) и не знает truth.
Если его ответ совпадает с эталоном, значит:
* расхождение реально видно в записях,
* доказательств достаточно,
* отвлекающие случаи действительно корректны (детектор их не флагует).
"""
import json
from collections import Counter, defaultdict

from .. import money
from .. import timeutil as tu
from .matcher import evidence_universe, match_findings, validate_submission


def open_data(ep) -> dict:
    return {"manifest": ep.manifest, "policy": ep.policy, "ledger": ep.ledger, "invoices": ep.invoices,
            "payments": ep.payments, "fx": ep.fx}


def _finding(kind, affected, amount, currency, evidence, text):
    return {"type": kind, "affected_ids": sorted(affected), "claimed_amount_minor": amount, "currency": currency,
            "evidence_ids": list(evidence), "explanation": text}


def detect(data: dict, naive: bool = False) -> list:
    """Находит все четыре типа ошибок. naive=True: упрощённая сверка для базовой политики
    (курс всегда по дате UTC и без сравнения состава line_items)."""
    manifest, policy = data["manifest"], data["policy"]
    company_tz = tu.tz_from_offset(manifest["timezone"])
    start, end = tu.parse_ts(manifest["period_start"]), tu.parse_ts(manifest["period_end"])
    rule = "utc_date" if naive else policy["fx_rate_date"]
    rounding = policy["rounding"]
    invoices = {i["invoice_id"]: i for i in data["invoices"]}
    rates = {(r["date"], r["source_currency"], r["target_currency"]): r["rate"] for r in data["fx"]}
    pay_rows, inv_rows = defaultdict(list), defaultdict(list)
    for row in data["ledger"]:
        if row["payment_id"]:
            pay_rows[row["payment_id"]].append(row)
        elif row["invoice_id"]:
            inv_rows[row["invoice_id"]].append(row)
    findings = []

    # 1. Дубли счетов
    groups = defaultdict(list)
    for inv in data["invoices"]:
        lines = None if naive else json.dumps(inv["line_items"], sort_keys=True)
        groups[(inv["entity_id"], inv["currency"], inv["total_minor"], lines)].append(inv)
    for group in groups.values():
        group.sort(key=lambda i: (tu.parse_ts(i["issued_at"]), i["invoice_id"]))
        for dup in group[1:]:
            orig = group[0]
            findings.append(_finding("duplicate_invoice", [orig["invoice_id"], dup["invoice_id"]], dup["total_minor"],
                                     dup["currency"], [orig["doc_id"], dup["doc_id"]],
                                     "Same client, currency, total and line items."))

    # 2-3. Распределение платежа и курс
    for pay in data["payments"]:
        refs = pay["invoice_refs"]
        if len(refs) != 1:
            continue  # аванс или платёж по нескольким счетам: ошибок такого вида не вносится
        ref = refs[0]
        leg = invoices[ref["invoice_id"]]["currency"]
        rows = pay_rows.get(pay["payment_id"], [])
        ar = [r for r in rows if r["account"] == "AR"]
        un = [r for r in rows if r["account"] == "UNAPPLIED"]
        net = lambda items: sum(r["credit_minor"] - r["debit_minor"] for r in items)
        unapplied, booked = net(un), net(ar) + net(un)
        correct, fx_key = ref["amount_minor"], None
        if leg != pay["currency"]:
            date = tu.fx_date(tu.parse_ts(pay["timestamp"]), rule, company_tz)
            fx_key = f"{date}:{pay['currency']}:{leg}"
            correct = money.convert(ref["amount_minor"], pay["currency"], leg, rates[(date, pay["currency"], leg)], rounding)
        if unapplied > 0:
            findings.append(_finding("payment_allocation", [pay["payment_id"]], unapplied, leg,
                                     [pay["doc_id"]] + [r["row_id"] for r in un], "Part of the payment sits in UNAPPLIED."))
        if fx_key and abs(booked - correct) > 1:
            findings.append(_finding("fx_error", [pay["payment_id"]], abs(booked - correct), leg,
                                     [pay["doc_id"]] + [r["row_id"] for r in ar + un] + [fx_key],
                                     "Booked amount differs from the policy-rate conversion."))

    # 4. Отнесение к периоду
    for inv in data["invoices"]:
        for row in (r for r in inv_rows.get(inv["invoice_id"], []) if r["account"] == "AR"):
            doc_side = tu.period_side(tu.parse_ts(inv["issued_at"]), start, end)
            row_side = tu.period_side(tu.parse_ts(row["timestamp"]), start, end)
            if doc_side != row_side:
                findings.append(_finding("period_cutoff", [inv["invoice_id"]], inv["total_minor"], inv["currency"],
                                         [inv["doc_id"], row["row_id"]], "Document and posting fall into different periods."))
    return findings


# ---------------------------------------------------------------- проверки эпизода

def _unique(values):
    return len(values) == len(set(values))


def _instant(text):
    return tu.parse_ts(text).astimezone(tu.timezone.utc)


def check_schema(ep) -> list:
    problems = []
    m = ep.manifest
    counts = {"ledger_rows": len(ep.ledger), "documents": len(ep.invoices) + len(ep.payments),
              "invoices": len(ep.invoices), "payments": len(ep.payments), "fx_rows": len(ep.fx)}
    if m["counts"] != counts:
        problems.append("manifest.counts do not match the files")
    if not 200 <= len(ep.ledger) <= 2000:
        problems.append(f"ledger size {len(ep.ledger)} is outside 200..2000")
    docs = len(ep.invoices) + len(ep.payments)
    if docs > 200:
        problems.append(f"{docs} documents is more than 200")
    ids = [r["row_id"] for r in ep.ledger]
    ids_docs = [d["doc_id"] for d in ep.invoices + ep.payments]
    if not (_unique(ids) and _unique(ids_docs) and _unique([i["invoice_id"] for i in ep.invoices])
            and _unique([p["payment_id"] for p in ep.payments])):
        problems.append("duplicate identifiers")
    inv_ids = {i["invoice_id"] for i in ep.invoices}
    pay_ids = {p["payment_id"] for p in ep.payments}
    for r in ep.ledger:
        for name in ("debit_minor", "credit_minor"):
            if isinstance(r[name], bool) or not isinstance(r[name], int) or r[name] < 0:
                problems.append(f"{r['row_id']}: {name} must be a non-negative integer")
        if r["debit_minor"] and r["credit_minor"]:
            problems.append(f"{r['row_id']}: debit and credit in one row")
        if r["currency"] not in m["currency_exponents"]:
            problems.append(f"{r['row_id']}: unknown currency")
        if r["invoice_id"] is not None and r["invoice_id"] not in inv_ids:
            problems.append(f"{r['row_id']}: dangling invoice_id")
        if r["payment_id"] is not None and r["payment_id"] not in pay_ids:
            problems.append(f"{r['row_id']}: dangling payment_id")
        try:
            tu.parse_ts(r["timestamp"])
        except ValueError:
            problems.append(f"{r['row_id']}: timestamp without offset")
    for rate in (f["rate"] for f in ep.fx):
        if not isinstance(rate, str):
            problems.append("fx rate must be a decimal string")
            break
    for p in ep.payments:
        for ref in p["invoice_refs"]:
            if ref["invoice_id"] not in inv_ids:
                problems.append(f"{p['payment_id']}: dangling invoice ref")
    return problems


def check_clean_books(ep) -> list:
    """Бухгалтерские равенства ДО внесения ошибок: проводки сбалансированы, оплат не больше счёта."""
    problems = []
    base, rule = ep.manifest["base_currency"], ep.policy["fx_rate_date"]
    company_tz = tu.tz_from_offset(ep.manifest["timezone"])
    rates = {(r["date"], r["source_currency"]): r["rate"] for r in ep.fx}
    groups = defaultdict(list)
    for row in ep.provenance["clean_ledger"]:
        if row["payment_id"]:
            groups[("pay", row["payment_id"])].append(row)
        elif row["invoice_id"]:
            groups[("inv", row["invoice_id"])].append(row)
        else:
            groups[("journal", row["timestamp"], row["currency"])].append(row)
    for key, rows in groups.items():
        if len({r["currency"] for r in rows}) == 1:
            if sum(r["debit_minor"] - r["credit_minor"] for r in rows) != 0:
                problems.append(f"clean group {key[:2]} is not balanced")
        else:  # разные валюты: приводим к базовой по курсу политики
            total = 0
            for r in rows:
                value = r["debit_minor"] - r["credit_minor"]
                if r["currency"] != base:
                    date = tu.fx_date(tu.parse_ts(r["timestamp"]), rule, company_tz)
                    value = money.convert(value, r["currency"], base, rates[(date, r["currency"])], ep.policy["rounding"])
                total += value
            if abs(total) > 1:
                problems.append(f"clean group {key[:2]} is not balanced in base currency")
    totals = {i["invoice_id"]: i["total_minor"] for i in ep.invoices}
    applied = Counter()
    for row in ep.provenance["clean_ledger"]:
        if row["account"] == "AR" and row["payment_id"]:
            applied[row["invoice_id"]] += row["credit_minor"] - row["debit_minor"]
    for invoice_id, value in applied.items():
        # допуск 1%: платёж в другой валюте округляется до минимальной единицы своей валюты
        slack = max(1, totals.get(invoice_id, 0) // 100)
        if invoice_id in totals and not 0 <= value <= totals[invoice_id] + slack:
            problems.append(f"{invoice_id}: applied payments {value} outside 0..total")
    return problems


def check_divergence(ep) -> list:
    """Фактические проводки отличаются от чистых только у объектов из эталона."""
    affected = {i for issue in ep.truth["issues"] for i in issue["affected_ids"]}

    def keep(row):
        return row["invoice_id"] not in affected and row["payment_id"] not in affected

    def signature(row):
        return (_instant(row["timestamp"]), row["account"], row["debit_minor"], row["credit_minor"], row["currency"],
                row["entity_id"], row["invoice_id"], row["payment_id"])

    actual = Counter(signature(r) for r in ep.ledger if keep(r))
    clean = Counter(signature(r) for r in ep.provenance["clean_ledger"] if keep(r))
    return [] if actual == clean else ["ledger differs from the clean ledger outside the declared issues"]


def check_truth(ep) -> list:
    """Эталон корректен: доказательства существуют, детектор по открытым файлам даёт тот же ответ."""
    problems = []
    universe = {r["row_id"] for r in ep.ledger} | {d["doc_id"] for d in ep.invoices + ep.payments}
    universe |= {f"{r['date']}:{r['source_currency']}:{r['target_currency']}" for r in ep.fx}
    seen = set()
    for issue in ep.truth["issues"]:
        key = (issue["type"], tuple(issue["affected_ids"]))
        if key in seen:
            problems.append(f"{issue['issue_id']}: duplicate truth key")
        seen.add(key)
        if issue["expected_amount_minor"] < 1:
            problems.append(f"{issue['issue_id']}: expected amount must be positive")
        if any(e not in universe for s in issue["evidence_sets"] for e in s):
            problems.append(f"{issue['issue_id']}: evidence refers to a record that does not exist")
    if len(ep.truth["issues"]) != len(ep.provenance["changes"]):
        problems.append("provenance.changes does not match truth.issues")
    found = validate_submission({"episode_id": ep.episode_id, "findings": detect(open_data(ep))}, ep.episode_id)
    result = match_findings(found, ep.truth["issues"], evidence_universe(open_data(ep)))
    if (result.fp, result.fn) != (0, 0):
        problems.append(f"independent detector disagrees with truth: FP={result.fp} FN={result.fn}")
    return problems


def check_episode(ep) -> list:
    """Все проверки данных. Пустой список = эпизод корректен."""
    return check_schema(ep) + check_clean_books(ep) + check_divergence(ep) + check_truth(ep)
