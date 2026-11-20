"""Простые трюки не должны работать: задача требует анализа доказательств (раздел 02 кейса).

Каждый тест повторяет эвристику, которую нашла «красная команда», и проверяет, что на наших данных
она больше не даёт высокой точности. Сравниваем с эталоном только для измерения, агент его не видит.
"""
from datetime import timedelta
from decimal import Decimal

import pytest

from audit_env.env import AuditEnv
from audit_env.generator import generate
from audit_env import timeutil as tu
from helpers import CLEAN, FULL, finding_from_issue, submit

SEEDS = range(1, 41)
MIX = [CLEAN, FULL, {"errors": {"period_cutoff": 2, "fx_error": 2}, "n_journal": 150}]


def episodes():
    for seed in SEEDS:
        for config in MIX:
            yield generate(seed, config)


def precision(flagged, truth):
    return len(flagged & truth) / len(flagged) if flagged else 1.0


def test_posting_delay_alone_does_not_reveal_cutoff_errors():
    """Трюк: «проводка AR отличается от issued_at больше чем на 15 минут = ошибка периода» (было P=1.0)."""
    flagged, truth = set(), set()
    for ep in episodes():
        truth |= {(ep.episode_id, i["affected_ids"][0]) for i in ep.truth["issues"] if i["type"] == "period_cutoff"}
        posted = {r["invoice_id"]: tu.parse_ts(r["timestamp"]) for r in ep.ledger
                  if r["account"] == "AR" and r["payment_id"] is None and r["invoice_id"]}
        for inv in ep.invoices:
            if abs(posted[inv["invoice_id"]] - tu.parse_ts(inv["issued_at"])) > timedelta(minutes=15):
                flagged.add((ep.episode_id, inv["invoice_id"]))
    assert truth <= flagged  # ошибки среди помеченных есть, но...
    assert precision(flagged, truth) < 0.15  # ...почти все помеченные законны


def test_ar_versus_invoice_total_does_not_reveal_fx_errors():
    """Трюк: «платёж в чужой валюте и кредит AR+UNAPPLIED отличается от суммы счёта > 0.3% = ошибка курса»."""
    flagged, truth = set(), set()
    for ep in episodes():
        truth |= {(ep.episode_id, i["affected_ids"][0]) for i in ep.truth["issues"] if i["type"] == "fx_error"}
        invoices = {i["invoice_id"]: i for i in ep.invoices}
        for pay in ep.payments:
            if len(pay["invoice_refs"]) != 1:
                continue
            inv = invoices[pay["invoice_refs"][0]["invoice_id"]]
            if inv["currency"] == pay["currency"]:
                continue
            booked = sum(r["credit_minor"] - r["debit_minor"] for r in ep.ledger
                         if r["payment_id"] == pay["payment_id"] and r["account"] in ("AR", "UNAPPLIED"))
            if abs(booked - inv["total_minor"]) > 0.003 * inv["total_minor"]:
                flagged.add((ep.episode_id, pay["payment_id"]))
    assert precision(flagged, truth) < 0.6


def test_fx_table_has_no_planted_spikes():
    """Раньше ошибка «не та дата» правила соседний курс в fx.csv, и в ряду курсов появлялся скачок."""
    def spikes(ep):
        series = {}
        for row in ep.fx:
            series.setdefault(row["source_currency"], []).append(Decimal(row["rate"]))
        count = 0
        for rates in series.values():
            for a, b, c in zip(rates, rates[1:], rates[2:]):
                up, down = b / a - 1, c / b - 1
                if abs(up) > Decimal("0.009") and abs(down) > Decimal("0.009") and up * down < 0:
                    count += 1
        return count
    with_fx = sum(spikes(generate(s, {"errors": {"fx_error": 3}, "n_journal": 150})) for s in SEEDS)
    clean = sum(spikes(generate(s, CLEAN)) for s in SEEDS)
    assert with_fx <= clean + 5  # скачков столько же, сколько в чистых компаниях


def test_planted_timestamps_are_not_round():
    """Раньше ошибки периода сдвигались на целые часы, и время было ровно HH:00:00."""
    round_error = round_other = total_error = total_other = 0
    for ep in episodes():
        cutoff = {i["affected_ids"][0] for i in ep.truth["issues"] if i["type"] == "period_cutoff"}
        for r in ep.ledger:
            if r["account"] != "AR" or r["payment_id"] is not None or not r["invoice_id"]:
                continue
            is_round = tu.parse_ts(r["timestamp"]).strftime("%M:%S") == "00:00"
            if r["invoice_id"] in cutoff:
                total_error += 1
                round_error += is_round
            else:
                total_other += 1
                round_other += is_round
    assert total_error > 0
    assert round_error / total_error < 0.05


def test_line_item_wording_does_not_separate_distractors():
    """Раньше тексты «Weekly support…», «… order N», «… job N-N» были только у отвлекающих счетов."""
    styles = {}
    for seed in SEEDS:
        ep = generate(seed, FULL)
        for inv in ep.invoices:
            for line in inv["line_items"]:
                text = line["description"]
                style = "week" if ", week " in text else "order" if " order " in text else "month"
                styles.setdefault(style, 0)
                styles[style] += 1
    assert {"week", "order", "month"} <= set(styles)  # все шаблоны встречаются у обычных счетов тоже


def test_listing_every_fx_date_as_evidence_is_not_accepted():
    """Трюк: вписать курсы на дату UTC и на дату компании, чтобы не выбирать по правилу политики."""
    env = AuditEnv()
    _, info = env.reset(3, FULL)
    truth = env.evaluator_truth()
    issue = next(i for i in truth["issues"] if i["type"] == "fx_error")
    fx_id = next(e for e in issue["evidence_sets"][0] if e.count(":") == 2)
    day, src, dst = fx_id.split(":")
    other_day = (tu.parse_ts(day + "T12:00:00+00:00") - timedelta(days=1)).date().isoformat()
    hedged = finding_from_issue(issue, evidence_ids=issue["evidence_sets"][0] + [f"{other_day}:{src}:{dst}"])
    env.step(submit(info["episode_id"], [hedged]))
    assert env.final_report()["tp"] == 0
    env.reset(3, FULL)
    unrelated = finding_from_issue(issue, evidence_ids=issue["evidence_sets"][0] + ["DOC-unrelated"])
    env.step(submit(info["episode_id"], [unrelated]))
    assert env.final_report()["tp"] == 1  # лишний посторонний ID не наказывается, противоречивый курс — да


@pytest.mark.parametrize("seed", range(1, 6))
def test_paired_variant_also_shuffles_the_ledger(seed):
    a = generate(seed, FULL)
    b = generate(seed, {**FULL, "variant": {"id_salt": "B", "shuffle": True}})
    times = lambda ep: [tu.parse_ts(r["timestamp"]) for r in ep.ledger]
    assert times(a) == sorted(times(a))
    assert times(b) != sorted(times(b))
