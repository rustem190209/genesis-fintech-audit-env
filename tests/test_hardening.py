"""Находки ревью кода: падения от чужого ввода, обходы проверок, секрет episode_id, границы config."""
import json
import subprocess
import sys
from decimal import Context, localcontext, setcontext, getcontext
from pathlib import Path

import pytest

from audit_env import money
from audit_env.calculator import CalcError, evaluate
from audit_env.env import AuditEnv
from audit_env.generator import generate, merge_config
from audit_env.grader import StructureError, validate_submission
from helpers import CLEAN, FULL, calc, finding_from_issue, submit

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env():
    e = AuditEnv()
    e.reset(1, FULL)
    return e


# ---- калькулятор не падает на больших числах

@pytest.mark.parametrize("expr", ["10000000000000000", "99999999999999999+1", "1" + "0" * 29, "999999999999999*999999999999999",
                                  "9" * 30 + "/0.000000000001"])
def test_calculator_large_results_are_errors_not_crashes(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_calculator_accepts_the_largest_allowed_result():
    assert evaluate("999999999999999+1") == "1000000000000000"
    assert evaluate("1000000000000000/3") == "333333333333333.333333333333"


def test_calculator_error_in_env_is_a_bounded_error_and_costs_a_step(env):
    obs = env.step(calc("10000000000000000"))[0]
    assert obs["error"]["code"] == "calc_error" and obs["step"] == 1


def test_calculator_ignores_a_changed_global_decimal_context():
    saved = getcontext().copy()
    try:
        setcontext(Context(prec=3))
        assert evaluate("1/3") == "0.333333333333"
    finally:
        setcontext(saved)


def test_money_and_generator_ignore_a_changed_global_decimal_context():
    expected = generate(2, FULL).open_files()
    saved = getcontext().copy()
    try:
        setcontext(Context(prec=5, rounding="ROUND_DOWN"))
        assert generate(2, FULL).open_files() == expected
        assert money.convert(1234567, "EUR", "USD", "1.08123456") == 1334857  # 12345.67 * 1.08123456 = 13348.57
    finally:
        setcontext(saved)


# ---- курсор

@pytest.mark.parametrize("cursor", ["²", "٣", "１２", "9" * 5000, "-1", "1.5", " 1", "", "1e3", 5, 1.0, [], {}, True])
def test_odd_cursors_are_bounded_errors(env, cursor):
    obs = env.step({"tool": "list_records", "args": {"table": "ledger", "cursor": cursor}})[0]
    assert obs["error"]["code"] == "invalid_cursor" and obs["step"] == 1


# ---- submit

@pytest.mark.parametrize("args", [{"findings": None}, {"findings": None, "x": 1}, {"findings": 5}, {"findings": "x"}, {"findings": []}])
def test_submit_payload_must_be_a_real_findings_object(args):
    env = AuditEnv()
    env.reset(1, CLEAN)
    obs, reward, terminated, _, _ = env.step({"tool": "submit", "args": args})
    assert terminated and reward == -1.0 and obs["error"]["code"] == "invalid_submission"


def test_null_findings_no_longer_scores_as_an_empty_answer():
    env = AuditEnv()
    env.reset(1, CLEAN)  # чистая компания: пустой ответ дал бы +1
    assert env.step({"tool": "submit", "args": {"findings": None}})[1] == -1.0


@pytest.mark.parametrize("change", [{"type": "x" * 33}, {"currency": "USDUSDUSD"}, {"affected_ids": ["I" * 65]},
                                    {"evidence_ids": ["E" * 65]}])
def test_long_strings_are_a_structure_error(change):
    env = AuditEnv()
    env.reset(1, FULL)
    issue = env.evaluator_truth()["issues"][0]
    payload = {"episode_id": env.evaluator_truth()["episode_id"], "findings": [finding_from_issue(issue, **change)]}
    with pytest.raises(StructureError):
        validate_submission(payload, payload["episode_id"])


# ---- записи нельзя испортить через выданные объекты

def test_mutating_returned_records_does_not_change_the_environment(env):
    first = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 3}})[0]["result"]["records"]
    first[0]["debit_minor"] = 987654321
    first[0]["account"] = "HACKED"
    again = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 3}})[0]["result"]["records"]
    assert again[0]["account"] != "HACKED" and again[0]["debit_minor"] != 987654321
    inv = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})[0]["result"]["records"][0]
    doc = env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]["result"]["document"]
    doc["line_items"].clear()
    assert env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]["result"]["document"]["line_items"]


def test_mutating_the_reset_observation_does_not_change_the_environment():
    env = AuditEnv()
    obs, info = env.reset(1, FULL)
    obs["policy"]["rounding"] = "hacked"
    obs["manifest"]["base_currency"] = "XXX"
    assert env.reset(1, FULL)[0]["policy"]["rounding"] != "hacked"


# ---- episode_id и секрет проверяющего

def test_episode_id_depends_on_the_secret_but_the_data_does_not(monkeypatch):
    monkeypatch.delenv("AUDIT_ENV_SECRET", raising=False)
    plain = generate(5, FULL)
    monkeypatch.setenv("AUDIT_ENV_SECRET", "organizer-secret")
    secret = generate(5, FULL)
    other = generate(6, FULL)
    assert plain.episode_id != secret.episode_id
    for name in ("ledger", "invoices", "payments", "fx"):
        assert getattr(plain, name) == getattr(secret, name)
    assert secret.truth["issues"] == [{**i} for i in secret.truth["issues"]]
    assert secret.episode_id != other.episode_id


def test_a_secret_stops_brute_forcing_the_seed_from_the_episode_id(monkeypatch):
    monkeypatch.setenv("AUDIT_ENV_SECRET", "hidden")
    target = generate(777, FULL).episode_id
    monkeypatch.delenv("AUDIT_ENV_SECRET")
    guesses = {generate(seed, FULL).episode_id for seed in range(760, 800)}  # агент перебирает seed без секрета
    assert target not in guesses


# ---- границы config

@pytest.mark.parametrize("bad", [
    [], "text", 5, {"interactions": None}, {"interactions": 5}, {"interactions": "dup_cutoff"},
    {"interactions": ["dup_cutoff", "dup_cutoff"]}, {"interactions": [["x"]]},
    {"errors": {"fx_error": True}}, {"errors": {"fx_error": 1.5}}, {"errors": {"fx_error": "1"}},
    {"distractors": {"split_payments": 10 ** 7}}, {"distractors": {"split_payments": 11}},
    {"distractors": {"refunds": True}}, {"distractors": {"refunds": -1}},
    {"n_invoices": 81}, {"n_invoices": 19}, {"n_invoices": True}, {"n_journal": 801}, {"n_journal": -1},
    {"variant": {"id_salt": "x" * 65, "shuffle": False}}, {"variant": {"id_salt": 5, "shuffle": False}},
    {"variant": {"id_salt": "", "shuffle": "yes"}}, {"variant": []}, {"errors": []}, {"distractors": "none"},
])
def test_bad_config_raises_value_error(bad):
    with pytest.raises(ValueError):
        merge_config(bad)


def test_config_that_gives_an_out_of_range_episode_is_rejected():
    with pytest.raises(ValueError):
        generate(1, {"n_invoices": 20, "n_journal": 0, "distractors": {k: 0 for k in generate(1, CLEAN).provenance["config"]["distractors"]}})
    with pytest.raises(ValueError):
        generate(1, {"n_invoices": 80, "n_journal": 0, "distractors": {k: 10 for k in generate(1, CLEAN).provenance["config"]["distractors"]}})


def test_equal_configs_with_true_and_one_no_longer_exist():
    # раньше True и 1 давали разные эпизоды при одинаковом смысле; теперь True запрещён
    with pytest.raises(ValueError):
        generate(1, {"errors": {"fx_error": True}})
    assert generate(1, {"errors": {"fx_error": 1}}).truth["issues"]


# ---- сервер не падает

def _server_lines(lines):
    proc = subprocess.run([sys.executable, "-m", "audit_env.server"], cwd=ROOT, input=lines, capture_output=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    return [json.loads(x) for x in proc.stdout.decode("utf-8").splitlines()]


def test_server_survives_malformed_input_and_keeps_serving():
    deep = ("[" * 100000 + "]" * 100000).encode()
    inputs = [b'{"cmd":"reset","seed":1,"config":[1,2]}', b'{"cmd":"reset","seed":1,"config":"x"}',
              b'{"cmd":"reset","seed":1,"config":{"interactions":null}}', b'{"cmd":"reset","seed":1,"config":{"errors":null}}',
              deep, b'\xff\xfe{"cmd":"reset"}', b'{"cmd":"step","action":{"tool":"calculator","args":{"expression":"1+1"}}}',
              b'{"cmd":"reset","seed":1}',
              b'{"cmd":"step","action":{"tool":"list_records","args":{"table":"ledger","cursor":"\xc2\xb2"}}}',
              b'{"cmd":"step","action":{"tool":"calculator","args":{"expression":"10000000000000000"}}}']
    replies = _server_lines(b"\n".join(inputs) + b"\n")
    assert len(replies) == len(inputs)
    assert all("error" in r for r in replies[:6])
    assert replies[6].get("error")  # шаг до reset: сообщение, а не падение
    assert replies[7]["observation"]["budget_remaining"] == 100
    assert replies[8]["observation"]["error"]["code"] == "invalid_cursor"
    assert replies[9]["observation"]["error"]["code"] == "calc_error"
