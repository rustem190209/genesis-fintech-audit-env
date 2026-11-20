"""Контракт reset/step: форма ответов, лимиты, завершение, неверный ввод."""
import pytest

from audit_env.env import MAX_STEPS, PAGE_MAX, TABLES, AuditEnv
from helpers import CLEAN, FULL, calc, submit


@pytest.fixture
def env():
    e = AuditEnv()
    e.reset(1, FULL)
    return e


def test_reset_returns_observation_and_info():
    obs, info = AuditEnv().reset(1, FULL)
    assert {"task", "manifest", "policy", "tables", "budget_remaining"} <= set(obs)
    assert obs["budget_remaining"] == MAX_STEPS and obs["step"] == 0
    assert info["episode_id"] == obs["manifest"]["episode_id"]
    assert set(obs["manifest"]) >= {"schema_version", "episode_id", "base_currency", "period_start", "period_end",
                                    "timezone", "counts", "rounding"}


def test_step_returns_gymnasium_tuple(env):
    out = env.step(calc())
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert (reward, terminated, truncated) == (0.0, False, False)
    assert obs["step"] == 1 and obs["budget_remaining"] == MAX_STEPS - 1 and obs["error"] is None
    assert obs["result"] == {"value": "2"}


def test_pages_are_at_most_50_and_cursors_are_stable(env):
    for table in TABLES:
        first = env.step({"tool": "list_records", "args": {"table": table}})[0]["result"]
        again = env.step({"tool": "list_records", "args": {"table": table}})[0]["result"]
        assert first == again
        assert len(first["records"]) <= PAGE_MAX


def test_walking_cursors_covers_each_row_once(env):
    obs0, _ = env.reset(2, FULL)
    for table in TABLES:
        seen, cursor = [], None
        while True:
            result = env.step({"tool": "list_records", "args": {"table": table, "cursor": cursor, "limit": 50}})[0]["result"]
            seen.extend(result["records"])
            cursor = result["next_cursor"]
            if cursor is None:
                break
        assert len(seen) == obs0["tables"][table]
        assert len({repr(r) for r in seen}) == len(seen)
        env.reset(2, FULL)


def test_limit_is_clipped_and_validated(env):
    big = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": 500}})[0]
    assert len(big["result"]["records"]) == PAGE_MAX
    for bad in (0, -3, "10", 2.5, True):
        obs = env.step({"tool": "list_records", "args": {"table": "ledger", "limit": bad}})[0]
        assert obs["error"]["code"] == "invalid_args"


def test_get_record_and_get_document(env):
    inv = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 1}})[0]["result"]["records"][0]
    assert "line_items" not in inv  # в списке состава нет: его надо запросить документом
    rec = env.step({"tool": "get_record", "args": {"table": "invoices", "id": inv["invoice_id"]}})[0]["result"]["record"]
    doc = env.step({"tool": "get_document", "args": {"doc_id": inv["doc_id"]}})[0]["result"]
    assert doc["doc_type"] == "invoice" and doc["document"]["line_items"] == rec["line_items"]


@pytest.mark.parametrize("action", [
    {"tool": "get_record", "args": {"table": "ledger", "id": "nope"}},
    {"tool": "get_record", "args": {"table": "bad", "id": "x"}},
    {"tool": "get_document", "args": {"doc_id": "nope"}},
    {"tool": "get_document", "args": {"doc_id": 5}},
    {"tool": "teleport", "args": {}},
    {"tool": "list_records", "args": {"table": "ledger", "cursor": "abc"}},
    {"tool": "list_records", "args": {"table": "ledger", "cursor": "99999999"}},
    {"tool": "calculator", "args": {"expression": "1/0"}},
    {"tool": "calculator", "args": {}},
    "not a dict", None, {"tool": 5}, {"tool": "calculator", "args": []}, {},
])
def test_bad_input_gives_bounded_error_and_costs_a_step(env, action):
    obs, reward, terminated, truncated, _ = env.step(action)
    assert obs["error"] and obs["result"] is None and obs["step"] == 1
    assert obs["budget_remaining"] == MAX_STEPS - 1
    assert len(obs["error"]["message"]) < 100
    assert (reward, terminated, truncated) == (0.0, False, False)


def test_error_message_does_not_echo_long_input(env):
    obs = env.step({"tool": "get_record", "args": {"table": "ledger", "id": "A" * 5000}})[0]
    assert len(obs["error"]["message"]) < 100


def test_step_limit_truncates_with_empty_answer():
    env = AuditEnv()
    env.reset(1, CLEAN)
    for i in range(MAX_STEPS):
        obs, reward, terminated, truncated, _ = env.step(calc())
    assert truncated and not terminated and obs["budget_remaining"] == 0
    report = env.final_report()
    assert report["truncated"] and report["steps"] == MAX_STEPS
    assert report["tp"] == 0 and report["fp"] == 0  # итоговый ответ пустой


def test_submit_on_the_last_step_is_terminated_not_truncated():
    env = AuditEnv()
    obs, info = env.reset(1, CLEAN)
    for _ in range(MAX_STEPS - 1):
        env.step(calc())
    _, _, terminated, truncated, _ = env.step(submit(info["episode_id"], []))
    assert terminated and not truncated


class TickClock:
    """Каждый вызов часов «стоит» tick секунд: так шаг среды занимает 2*tick (начало и конец шага)."""

    def __init__(self, tick):
        self.t, self.tick = 0.0, tick

    def __call__(self):
        value = self.t
        self.t += self.tick
        return value


def test_time_limit_truncates():
    env = AuditEnv(clock=TickClock(35.0))  # шаг занимает 35 с: после двух шагов набегает 70 > 60
    env.reset(1, FULL)
    assert env.step(calc())[3] is False
    assert env.step(calc())[3] is False
    obs, _, terminated, truncated, _ = env.step(calc())
    assert truncated and not terminated and obs["error"]["code"] == "time_limit"
    assert env.final_report()["tp"] == 0


def test_time_limit_counts_only_environment_time_not_agent_latency():
    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()
    env = AuditEnv(clock=clock)
    env.reset(1, FULL)
    for _ in range(10):
        clock.t += 100.0  # агент «думает» 100 секунд между шагами: на среду это не влияет
        obs, _, terminated, truncated, _ = env.step(calc())
        assert not truncated and obs["error"] is None


def test_submit_terminates_and_further_steps_raise():
    env = AuditEnv()
    _, info = env.reset(1, CLEAN)
    obs, _, terminated, _, _ = env.step(submit(info["episode_id"], []))
    assert terminated and obs["result"] == {"status": "submitted"}
    with pytest.raises(RuntimeError):
        env.step(calc())


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        AuditEnv().step(calc())


def test_final_report_only_after_done(env):
    with pytest.raises(RuntimeError):
        env.final_report()


def test_reset_isolates_episode_state():
    env = AuditEnv()
    env.reset(1, FULL)
    for _ in range(7):
        env.step(calc())
    obs_b, info_b = env.reset(2, FULL)
    fresh_obs, fresh_info = AuditEnv().reset(2, FULL)
    assert obs_b == fresh_obs and info_b == fresh_info
    assert env.step(calc())[0]["step"] == 1  # счётчик шагов сброшен


def test_same_seed_gives_same_reset():
    a, _ = AuditEnv().reset(5, FULL)
    b, _ = AuditEnv().reset(5, FULL)
    assert a == b


def test_invalid_config_is_rejected():
    for bad in ({"nope": 1}, {"errors": {"nope": 1}}, {"n_invoices": 5}, {"base_currency": "XXX"},
                {"errors": {"fx_error": -1}}, {"errors": {"fx_error": 13}}):
        with pytest.raises(ValueError):
            AuditEnv().reset(1, bad)
    with pytest.raises(ValueError):
        AuditEnv().reset("1", FULL)
