"""Изоляция: эталон не утекает через наблюдения, ошибки, info и журналы. Оценщик вне процесса агента."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from audit_env.env import TABLES, AuditEnv
from audit_env.generator import LEDGER_COLUMNS
from helpers import FULL, calc, dump, oracle_findings, submit

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_WORDS = ["ISS-", "truth", "provenance", "expected_amount", "evidence_sets", "tolerance_minor", "dup_of",
                   "clean_ledger", "planted", "\"plan\"", "wrong_rate", "\"tag\"", "interaction", "seed"]
SCHEMA = {
    "ledger": set(LEDGER_COLUMNS),
    "invoices": {"doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency", "total_minor"},
    "payments": {"doc_id", "payment_id", "timestamp", "currency", "amount_minor", "invoice_refs"},
    "fx": {"date", "source_currency", "target_currency", "rate"},
}


def read_everything(env, obs):
    """Читает все таблицы, все документы и провоцирует ошибки. Возвращает все наблюдения."""
    seen = [obs]
    for table in TABLES:
        cursor = None
        while True:
            out = env.step({"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}})
            seen.append(out)
            if env._state["step"] >= 99:
                return seen
            cursor = out[0]["result"]["next_cursor"]
            if cursor is None:
                break
    return seen


def test_observations_contain_no_evaluator_data():
    env = AuditEnv()
    obs, info = env.reset(3, FULL)
    outputs = read_everything(env, (obs, 0.0, False, False, info))
    outputs.append(env.step({"tool": "get_record", "args": {"table": "ledger", "id": "x"}}))
    outputs.append(env.step({"tool": "nonsense"}))
    text = dump(outputs)
    for word in FORBIDDEN_WORDS:
        assert word not in text, word


def test_record_fields_match_the_public_schema():
    env = AuditEnv()
    env.reset(3, FULL)
    for table in TABLES:
        rows = env.step({"tool": "list_records", "args": {"table": table, "limit": 50}})[0]["result"]["records"]
        assert all(set(r) == SCHEMA[table] for r in rows), table


def test_info_and_steps_never_carry_diagnostics():
    env = AuditEnv(mode="closed")
    obs, info = env.reset(3, FULL)
    assert set(info) == {"episode_id"}
    out = env.step(calc())
    assert set(out[4]) == {"episode_id", "step"} and set(out[0]) == {"step", "budget_remaining", "result", "error"}


def test_closed_mode_hides_reward_until_the_harness_reads_the_report():
    env = AuditEnv(mode="closed")
    _, info = env.reset(3, FULL)
    truth = env.evaluator_truth()
    obs, reward, terminated, _, info = env.step(submit(info["episode_id"], oracle_findings(truth)))
    assert terminated and reward == 0.0
    assert set(obs) == {"step", "budget_remaining", "result", "error"} and obs["result"] == {"status": "submitted"}
    assert env.final_report()["reward"] == 1.0  # награда есть, но только у проверяющего


def test_closed_mode_reward_is_zero_on_every_step_including_invalid_submit():
    env = AuditEnv(mode="closed")
    env.reset(3, FULL)
    out = env.step({"tool": "submit", "args": {"findings": {"bad": 1}}})
    assert out[1] == 0.0 and out[2] is True
    assert out[0]["error"]["code"] == "invalid_submission" and "episode_id" not in out[0]["error"]["message"]
    assert env.final_report()["reward"] == -1.0


def test_open_mode_returns_only_the_reward_not_metrics():
    env = AuditEnv(mode="open")
    _, info = env.reset(3, FULL)
    out = env.step(submit(info["episode_id"], []))
    assert out[1] < 0 and set(out[4]) == {"episode_id", "step"}


def test_guessing_ids_does_not_pay_off():
    env = AuditEnv()
    _, info = env.reset(3, FULL)
    truth = env.evaluator_truth()
    # правильные тип, сумма и валюта, но ID выдуманы и доказательств нет
    guesses = [{"type": i["type"], "affected_ids": ["INV-00000"], "claimed_amount_minor": i["expected_amount_minor"],
                "currency": i["currency"], "evidence_ids": [], "explanation": "guess"} for i in truth["issues"]]
    env.step(submit(info["episode_id"], guesses))
    assert env.final_report()["tp"] == 0


def test_truth_is_not_in_the_open_files():
    from audit_env.generator import generate
    ep = generate(3, FULL)
    text = "".join(ep.open_files().values())
    for word in FORBIDDEN_WORDS[:-1]:
        assert word not in text, word


def _talk(proc, message):
    proc.stdin.write(json.dumps(message) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def _server(*flags):
    return subprocess.Popen([sys.executable, "-m", "audit_env.server", *flags], cwd=ROOT, text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def test_out_of_process_server_keeps_the_evaluator_away_from_the_agent():
    proc = _server("--mode", "closed")
    try:
        reset = _talk(proc, {"cmd": "reset", "seed": 3, "config": FULL})
        assert reset["observation"]["budget_remaining"] == 100
        step = _talk(proc, {"cmd": "step", "action": calc("2*21")})
        assert step["observation"]["result"] == {"value": "42"}
        submitted = _talk(proc, {"cmd": "step", "action": submit(reset["info"]["episode_id"], [])})
        assert submitted["terminated"] and submitted["reward"] == 0.0
        assert "error" in _talk(proc, {"cmd": "report"})  # агент отчёт не получит
        assert "error" in _talk(proc, {"cmd": "step", "action": calc()})  # эпизод окончен, но сервер жив
        proc.stdin.write("not json\n")
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline()) == {"error": "invalid JSON"}
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


def test_harness_flag_exposes_the_report():
    proc = _server("--mode", "closed", "--harness")
    try:
        reset = _talk(proc, {"cmd": "reset", "seed": 3, "config": FULL})
        _talk(proc, {"cmd": "step", "action": submit(reset["info"]["episode_id"], [])})
        report = _talk(proc, {"cmd": "report"})["report"]
        assert report["status"] == "graded" and report["fn"] > 0
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)


@pytest.mark.parametrize("bad", ['{"cmd": "reset", "seed": "x"}', '[1,2]', '{"nocmd": 1}', '{"cmd": "fly"}'])
def test_server_survives_bad_commands(bad):
    proc = _server()
    try:
        proc.stdin.write(bad + "\n")
        proc.stdin.flush()
        assert "error" in json.loads(proc.stdout.readline())
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
