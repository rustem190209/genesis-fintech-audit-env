"""Генератор: воспроизводимость, корректность данных и эталона, чистые случаи, парные проверки."""
import copy

import pytest

from audit_env.files import load_open, write_episode
from audit_env.generator import ERROR_TYPES, generate
from audit_env.grader.validate import (check_clean_books, check_divergence, check_episode, check_schema, check_truth,
                                       detect, open_data)
from helpers import CLEAN, FULL, TYPES, oracle_findings, single

SEEDS = range(1, 13)


def types_of(ep):
    return sorted(i["type"] for i in ep.truth["issues"])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("config", [CLEAN, FULL, *(single(t, 2) for t in TYPES)], ids=["clean", "full", *TYPES])
def test_every_episode_passes_all_data_checks(seed, config):
    assert check_episode(generate(seed, config)) == []


def test_same_seed_config_version_gives_identical_files():
    a, b = generate(7, FULL), generate(7, copy.deepcopy(FULL))
    assert a.open_files() == b.open_files() and a.closed_files() == b.closed_files()


def test_different_seeds_give_different_episodes():
    assert generate(1, FULL).open_files() != generate(2, FULL).open_files()
    assert generate(1, FULL).episode_id != generate(2, FULL).episode_id


def test_config_changes_the_episode():
    assert generate(1, FULL).open_files() != generate(1, CLEAN).open_files()


@pytest.mark.parametrize("error_type", TYPES)
def test_each_error_type_is_independently_configurable(error_type):
    ep = generate(4, single(error_type, 2))
    assert types_of(ep) == [error_type] * 2


def test_counts_match_the_config():
    cfg = {"errors": {"duplicate_invoice": 2, "payment_allocation": 1, "fx_error": 3, "period_cutoff": 2}, "n_journal": 100}
    assert len(generate(5, cfg).truth["issues"]) == 8
    assert len(generate(5, FULL).truth["issues"]) == 4 + 4  # 4 типа + 2 сценария по 2 ошибки


def test_interacting_scenarios_produce_two_errors_each():
    ep = generate(6, {"interactions": ["fx_boundary_partial"], "n_journal": 100})
    assert types_of(ep) == ["fx_error", "payment_allocation"]
    same_payment = {tuple(i["affected_ids"]) for i in ep.truth["issues"]}
    assert len(same_payment) == 1  # обе ошибки на одном платеже
    ep = generate(6, {"interactions": ["dup_cutoff"], "n_journal": 100})
    assert types_of(ep) == ["duplicate_invoice", "period_cutoff"]


def test_max_twelve_issues_allowed_thirteen_rejected():
    assert len(generate(1, {"errors": {t: 3 for t in ERROR_TYPES}, "n_journal": 100}).truth["issues"]) == 12
    with pytest.raises(ValueError):
        generate(1, {"errors": {t: 3 for t in ERROR_TYPES}, "interactions": ["dup_cutoff"]})


@pytest.mark.parametrize("seed", SEEDS)
def test_clean_company_has_no_truth_and_distractors_are_not_flagged(seed):
    ep = generate(seed, CLEAN)
    assert ep.truth["issues"] == [] and detect(open_data(ep)) == []
    tags = ep.provenance["history"]["distractor_tags"]
    assert {"split", "refund", "recurring", "deposit", "tz_boundary", "fx_clean", "same_amount", "multi"} <= set(tags)


def test_distractors_can_be_switched_off_and_on():
    none = {k: 0 for k in generate(1, CLEAN).provenance["config"]["distractors"]}
    ep = generate(1, {"distractors": none, "n_journal": 150})
    assert ep.provenance["history"]["distractor_tags"] == []


@pytest.mark.parametrize("seed", SEEDS)
def test_naive_detector_is_wrong_in_the_expected_way(seed):
    """Упрощённая сверка ловит распределение платежей, но ошибается на отвлекающих случаях."""
    ep = generate(seed, CLEAN)
    assert len(detect(open_data(ep), naive=True)) > 0


def test_oracle_style_answer_always_scores_perfectly():
    from audit_env.grader import match_findings, validate_submission
    for seed in SEEDS:
        ep = generate(seed, FULL)
        result = match_findings(validate_submission({"episode_id": ep.episode_id, "findings": oracle_findings(ep.truth)},
                                                    ep.episode_id), ep.truth["issues"])
        assert (result.tp, result.fp, result.fn) == (len(ep.truth["issues"]), 0, 0)


def test_data_formats():
    ep = generate(3, FULL)
    assert ep.manifest["period_start"].endswith(ep.manifest["timezone"])
    assert all(isinstance(r["rate"], str) for r in ep.fx)
    assert all(isinstance(r["debit_minor"], int) and isinstance(r["credit_minor"], int) for r in ep.ledger)
    assert all(r["invoice_id"] is None or isinstance(r["invoice_id"], str) for r in ep.ledger)
    assert any(r["invoice_id"] is None for r in ep.ledger) and any(r["payment_id"] is None for r in ep.ledger)
    assert ep.open_files()["ledger.csv"].splitlines()[0] == "row_id,timestamp,account,debit_minor,credit_minor,currency,entity_id,invoice_id,payment_id"
    assert "null" in ep.open_files()["ledger.csv"]
    assert len({t[-6:] for t in (r["timestamp"] for r in ep.ledger)}) > 2  # разные смещения времени


def test_variety_across_seeds():
    eps = [generate(s, FULL) for s in range(1, 25)]
    assert len({e.manifest["base_currency"] for e in eps}) >= 2
    assert len({e.manifest["timezone"] for e in eps}) >= 3
    assert len({e.policy["fx_rate_date"] for e in eps}) == 2
    assert len({e.policy["rounding"] for e in eps}) == 2
    currencies = {r["currency"] for e in eps for r in e.invoices}
    assert len(currencies) >= 4
    assert len({i["currency"] for e in eps for i in e.truth["issues"]}) >= 2


def test_sizes_stay_within_the_proposed_limits():
    for n_journal in (50, 400, 700):
        ep = generate(2, {**FULL, "n_journal": n_journal})
        assert 200 <= len(ep.ledger) <= 2000
        assert len(ep.invoices) + len(ep.payments) <= 200
        assert check_schema(ep) == []


# ---------------- парные проверки: другой порядок и другие ID, та же история

PAIR = {"variant": {"id_salt": "B", "shuffle": True}}


@pytest.mark.parametrize("seed", range(1, 9))
def test_paired_variant_keeps_the_history_but_changes_ids_and_order(seed):
    a, b = generate(seed, FULL), generate(seed, {**FULL, **PAIR})
    assert check_episode(b) == []
    sig = lambda ep: sorted((i["type"], i["expected_amount_minor"], i["currency"], i["tolerance_minor"]) for i in ep.truth["issues"])
    assert sig(a) == sig(b)
    ids = lambda ep: {i["invoice_id"] for i in ep.invoices} | {p["payment_id"] for p in ep.payments}
    assert len(ids(a) & ids(b)) < 0.1 * len(ids(a))  # ID другие (редкие случайные совпадения возможны)
    assert [i["invoice_id"] for i in a.invoices] != [i["invoice_id"] for i in b.invoices]
    totals = lambda ep: sorted((i["currency"], i["total_minor"]) for i in ep.invoices)
    assert totals(a) == totals(b)  # те же суммы и валюты счетов
    assert a.episode_id != b.episode_id
    assert a.manifest["counts"] == b.manifest["counts"]


def test_paired_variant_gives_the_same_oracle_reward():
    from audit_env.runner import run_episode
    for seed in (1, 2, 3):
        ra, rb = run_episode("oracle", seed, FULL), run_episode("oracle", seed, {**FULL, **PAIR})
        assert ra["reward"] == rb["reward"] == 1.0 and ra["tp"] == rb["tp"]


# ---------------- проверки данных действительно ловят испорченные эпизоды

def test_validator_catches_a_tampered_ledger():
    ep = generate(2, FULL)
    affected = {i for issue in ep.truth["issues"] for i in issue["affected_ids"]}
    row = next(r for r in ep.ledger if r["invoice_id"] not in affected and r["payment_id"] not in affected and r["invoice_id"])
    row["debit_minor"] += 5  # правим строку, которой нет в эталоне
    assert check_divergence(ep) != []


def test_validator_catches_a_wrong_truth():
    ep = generate(2, FULL)
    ep.truth["issues"][0]["expected_amount_minor"] += 10 ** 6
    assert any("detector disagrees" in p for p in check_truth(ep))
    ep = generate(2, FULL)
    ep.truth["issues"][0]["evidence_sets"][0].append("DOC-NONEXISTENT")
    assert any("does not exist" in p for p in check_truth(ep))
    ep = generate(2, FULL)
    ep.truth["issues"].append({**ep.truth["issues"][0], "issue_id": "ISS-99"})
    assert check_truth(ep)


def test_validator_catches_unbalanced_clean_books():
    ep = generate(2, FULL)
    ep.provenance["clean_ledger"][0]["debit_minor"] += 7
    assert check_clean_books(ep) != []


def test_validator_catches_schema_problems():
    ep = generate(2, FULL)
    ep.ledger[0]["payment_id"] = "PAY-ghost"
    ep.ledger[1]["timestamp"] = "2026-03-01T10:00:00"
    assert len(check_schema(ep)) >= 2


def test_files_roundtrip(tmp_path):
    ep = generate(4, FULL)
    folder = write_episode(ep, tmp_path / "ep", with_closed=True)
    loaded = load_open(folder)
    assert loaded["ledger"] == ep.ledger and loaded["invoices"] == ep.invoices
    assert loaded["payments"] == ep.payments and loaded["fx"] == ep.fx and loaded["manifest"] == ep.manifest
    assert (folder / "evaluator" / "truth.json").exists() and not (folder / "truth.json").exists()
    assert detect(loaded) == detect(open_data(ep))
    assert write_episode(ep, tmp_path / "ep2") and not (tmp_path / "ep2" / "evaluator").exists()


# ---------------- явные варианты ошибок (жюри: «хотя бы один пропуск и один частичный случай» и т. п.)

@pytest.mark.parametrize("error_type,mode,field", [
    ("payment_allocation", "missing", "mode"), ("payment_allocation", "partial", "mode"),
    ("fx_error", "wrong_date", "fx_variant"), ("fx_error", "wrong_rate", "fx_variant"),
    ("period_cutoff", "end_late", "variant"), ("period_cutoff", "end_early", "variant"),
    ("period_cutoff", "start_late", "variant"), ("period_cutoff", "start_early", "variant"),
])
def test_error_modes_can_be_requested_explicitly(error_type, mode, field):
    for seed in (1, 2, 3):
        ep = generate(seed, {"errors": {error_type: 2}, "error_modes": {error_type: mode}, "n_journal": 150})
        assert check_episode(ep) == []
        changes = ep.provenance["changes"]
        values = [c["plan"][field] if field in ("mode", "fx_variant") else c[field] for c in changes]
        assert values == [mode, mode]


def test_unknown_error_mode_is_rejected():
    with pytest.raises(ValueError):
        generate(1, {"error_modes": {"fx_error": "sometimes"}})
