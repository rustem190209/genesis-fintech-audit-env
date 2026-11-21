"""Matcher, награда и атакующие ответы."""
import copy

import pytest

from audit_env.env import AuditEnv
from audit_env.grader import (INVALID_REWARD, StructureError, f1_score, match_findings, terminal_reward,
                              validate_submission)
from helpers import CLEAN, FULL, calc, finding_from_issue, oracle_findings, submit


def play(seed, config, findings, inspection=0, payload=None):
    """Играем эпизод: inspection пустых шагов, затем submit. Возвращает отчёт оценщика."""
    env = AuditEnv()
    _, info = env.reset(seed, config)
    for _ in range(inspection):
        env.step(calc())
    action = submit(info["episode_id"], findings)
    if payload is not None:
        action = {"tool": "submit", "args": {"findings": payload}}
    env.step(action)
    return env.final_report(), env.evaluator_truth()


def truth_of(seed=3, config=FULL):
    env = AuditEnv()
    env.reset(seed, config)
    return env.evaluator_truth()


# ---------------- формулы

@pytest.mark.parametrize("tp,fp,fn,expected", [(0, 0, 0, 1.0), (0, 3, 0, 0.0), (0, 0, 4, 0.0), (4, 0, 0, 1.0),
                                               (2, 1, 1, 2 * 2 / (4 + 1 + 1)), (1, 1, 0, 2 / 3)])
def test_f1(tp, fp, fn, expected):
    assert f1_score(tp, fp, fn) == pytest.approx(expected)


def test_reward_formula_and_clip():
    assert terminal_reward(1.0, 0) == 1.0
    assert terminal_reward(1.0, 10) == pytest.approx(0.98)
    assert terminal_reward(0.0, 0) == -1.0
    assert terminal_reward(0.0, 50) == -1.0  # нижняя граница
    assert terminal_reward(0.5, 20) == pytest.approx(-0.04)


def test_precision_and_recall_are_zero_on_zero_denominators():
    from audit_env.grader.matcher import MatchResult
    empty = MatchResult(tp=0, fp=0, fn=0)
    assert (empty.precision, empty.recall, empty.f1) == (0.0, 0.0, 1.0)
    assert (MatchResult(0, 2, 0).precision, MatchResult(0, 2, 0).recall) == (0.0, 0.0)


# ---------------- обязательные свойства награды из раздела 05

def test_full_correct_answer_beats_silence_on_faulty_books():
    truth = truth_of()
    good, _ = play(3, FULL, oracle_findings(truth), inspection=30)
    silent, _ = play(3, FULL, [], inspection=30)
    assert good["reward"] > silent["reward"] and good["reward"] == pytest.approx(0.94) and silent["reward"] == -1.0


def test_silence_beats_false_accusations_on_clean_books():
    silent, _ = play(3, CLEAN, [], inspection=20)
    fake = [{"type": "fx_error", "affected_ids": ["PAY-1"], "claimed_amount_minor": 100, "currency": "USD",
             "evidence_ids": [], "explanation": "x"}]
    accused, _ = play(3, CLEAN, fake, inspection=20)
    assert silent["reward"] == pytest.approx(0.96) and accused["reward"] == -1.0


def test_duplicate_findings_give_no_benefit():
    truth = truth_of()
    once = oracle_findings(truth)
    twice = once + copy.deepcopy(once)
    a, _ = play(3, FULL, once)
    b, _ = play(3, FULL, twice)
    assert b["tp"] == a["tp"] and b["fp"] == len(once) and b["reward"] < a["reward"]


def test_repeating_one_correct_finding_is_a_false_positive():
    truth = truth_of()
    f = finding_from_issue(truth["issues"][0])
    report, _ = play(3, FULL, [f, copy.deepcopy(f)])
    assert (report["tp"], report["fp"]) == (1, 1)


def test_flag_everything_scores_badly():
    from audit_env.runner import run_episode
    for seed in (1, 2, 3):
        assert run_episode("flag_all", seed, FULL)["reward"] <= -0.9
        assert run_episode("flag_all", seed, CLEAN)["reward"] == -1.0


def test_flagging_all_types_for_all_ids_even_with_correct_evidence_stays_bad():
    env = AuditEnv()
    _, info = env.reset(3, FULL)
    truth = env.evaluator_truth()
    findings = oracle_findings(truth)
    for issue in truth["issues"]:  # правильные ответы + по 3 лишних вывода на каждую ошибку
        for other in ("fx_error", "payment_allocation", "period_cutoff", "duplicate_invoice"):
            if other != issue["type"]:
                findings.append(finding_from_issue(issue, type=other))
    env.step(submit(info["episode_id"], findings[:30]))
    report = env.final_report()
    assert report["tp"] == len(truth["issues"]) and report["precision"] < 0.4 and report["reward"] < 0.2


def test_step_savings_only_slightly_separate_equal_answers():
    truth = truth_of()
    fast, _ = play(3, FULL, oracle_findings(truth), inspection=10)
    slow, _ = play(3, FULL, oracle_findings(truth), inspection=60)
    assert 0 < fast["reward"] - slow["reward"] == pytest.approx(0.1)
    wrong_fast, _ = play(3, FULL, [], inspection=0)
    right_slow, _ = play(3, FULL, oracle_findings(truth), inspection=90)
    assert right_slow["reward"] > wrong_fast["reward"]  # дешёвый неверный ответ не выигрывает у верного


# ---------------- условия совпадения

@pytest.fixture(scope="module")
def truth():
    return truth_of()


def one(truth, kind):
    return next(i for i in truth["issues"] if i["type"] == kind)


def grade_one(truth, **override):
    issue = one(truth, override.pop("kind", "period_cutoff"))
    report, _ = play(3, FULL, [finding_from_issue(issue, **override)])
    return report["tp"] == 1


def test_matching_conditions(truth):
    assert grade_one(truth)
    assert not grade_one(truth, currency="XXX")
    assert not grade_one(truth, evidence_ids=[])
    assert not grade_one(truth, evidence_ids=["DOC-ghost"])
    assert not grade_one(truth, affected_ids=["INV-ghost"])
    assert not grade_one(truth, type="fx_error")
    assert not grade_one(truth, type="made_up_type")


def test_amount_tolerance_is_inclusive(truth):
    issue = one(truth, "period_cutoff")
    tol, exact = issue["tolerance_minor"], issue["expected_amount_minor"]
    assert grade_one(truth, claimed_amount_minor=exact + tol)
    assert grade_one(truth, claimed_amount_minor=exact - tol)
    assert not grade_one(truth, claimed_amount_minor=exact + tol + 1)
    assert not grade_one(truth, claimed_amount_minor=0)


def test_fields_are_compared_exactly(truth):
    """Определения жюри: ID, type и currency сравниваются точно, без регистра и обрезки пробелов."""
    issue = one(truth, "period_cutoff")
    assert not grade_one(truth, type=" period_cutoff ")
    assert not grade_one(truth, type="Period_Cutoff")
    assert not grade_one(truth, currency=issue["currency"].lower())
    assert not grade_one(truth, currency=issue["currency"] + " ")
    assert not grade_one(truth, affected_ids=[f" {i}" for i in issue["affected_ids"]])
    assert grade_one(truth, affected_ids=issue["affected_ids"] * 3)  # повторы внутри списка: то же множество


def test_extra_existing_evidence_is_fine_but_unknown_evidence_is_not(truth):
    issue = one(truth, "period_cutoff")
    other = one(truth, "duplicate_invoice")["evidence_sets"][0]  # существующие документы другого счёта
    assert grade_one(truth, evidence_ids=issue["evidence_sets"][0] + other[:1])
    assert grade_one(truth, evidence_ids=issue["evidence_sets"][0] + issue["evidence_sets"][0])
    assert not grade_one(truth, evidence_ids=issue["evidence_sets"][0] + ["DOC-does-not-exist"])


def test_any_listed_evidence_set_is_enough(truth):
    issue = one(truth, "period_cutoff")
    assert len(issue["evidence_sets"]) == 2
    assert grade_one(truth, evidence_ids=issue["evidence_sets"][1])


def test_duplicate_affected_ids_order_does_not_matter(truth):
    issue = one(truth, "duplicate_invoice")
    assert grade_one(truth, kind="duplicate_invoice", affected_ids=list(reversed(issue["affected_ids"])))
    assert not grade_one(truth, kind="duplicate_invoice", affected_ids=issue["affected_ids"][:1])


def test_interacting_errors_are_scored_separately(truth):
    by_payment = {}
    for issue in truth["issues"]:
        if issue["type"] in ("fx_error", "payment_allocation"):
            by_payment.setdefault(tuple(issue["affected_ids"]), {})[issue["type"]] = issue
    pair = next(v for v in by_payment.values() if len(v) == 2)  # взаимодействующий платёж: обе ошибки на нём
    fx, alloc = pair["fx_error"], pair["payment_allocation"]
    report, _ = play(3, FULL, [finding_from_issue(fx)])
    assert (report["tp"], report["fp"], report["fn"]) == (1, 0, len(truth["issues"]) - 1)  # одной ошибки мало
    report, _ = play(3, FULL, [finding_from_issue(fx), finding_from_issue(alloc)])
    assert (report["tp"], report["fp"], report["fn"]) == (2, 0, len(truth["issues"]) - 2)


def test_first_fully_valid_finding_takes_the_match(truth):
    issue = one(truth, "period_cutoff")
    near_miss = finding_from_issue(issue, claimed_amount_minor=issue["expected_amount_minor"] + 10 ** 7)
    correct = finding_from_issue(issue)
    report, _ = play(3, FULL, [near_miss, correct])
    assert (report["tp"], report["fp"]) == (1, 1)
    report, _ = play(3, FULL, [correct, near_miss])
    assert (report["tp"], report["fp"]) == (1, 1)


def test_truth_issue_ids_are_unique_and_canonical(truth):
    keys = [(i["type"], tuple(i["affected_ids"])) for i in truth["issues"]]
    assert len(set(keys)) == len(keys)
    assert all(i["affected_ids"] == sorted(i["affected_ids"]) for i in truth["issues"])
    assert [i["issue_id"] for i in truth["issues"]] == [f"ISS-{n:02d}" for n in range(1, len(keys) + 1)]


# ---------------- неверная структура = награда -1

def good_payload(truth, episode_id):
    return {"episode_id": episode_id, "findings": oracle_findings(truth)}


def structure_cases(truth, episode_id):
    base = finding_from_issue(truth["issues"][0])
    wrap = lambda *f: {"episode_id": episode_id, "findings": list(f)}
    drop = lambda key: {k: v for k, v in base.items() if k != key}
    return {
        "not an object": [1, 2],
        "missing episode_id": {"findings": []},
        "wrong episode_id": {"episode_id": "ep-other", "findings": []},
        "findings not a list": {"episode_id": episode_id, "findings": {}},
        "31 findings": wrap(*[base] * 31),
        "missing key": wrap(drop("currency")),
        "float amount": wrap({**base, "claimed_amount_minor": 10.5}),
        "bool amount": wrap({**base, "claimed_amount_minor": True}),
        "string amount": wrap({**base, "claimed_amount_minor": "10"}),
        "ids not a list": wrap({**base, "affected_ids": "INV-1"}),
        "id not a string": wrap({**base, "evidence_ids": [5]}),
        "11 evidence ids": wrap({**base, "evidence_ids": [f"E{i}" for i in range(11)]}),
        "11 evidence ids with repeats": wrap({**base, "evidence_ids": ["E1"] * 11}),
        "explanation too long": wrap({**base, "explanation": "x" * 1001}),
        "finding not an object": wrap("text"),
    }


def test_invalid_structure_ends_the_episode_with_minus_one(truth):
    env = AuditEnv()
    _, info = env.reset(3, FULL)
    for name, payload in structure_cases(truth, info["episode_id"]).items():
        env.reset(3, FULL)
        obs, reward, terminated, _, _ = env.step({"tool": "submit", "args": {"findings": payload}})
        assert terminated and reward == INVALID_REWARD, name
        assert obs["error"]["code"] == "invalid_submission", name
        report = env.final_report()
        assert report["status"] == "invalid" and report["evaluator_log"], name  # ошибка в журнале оценщика


@pytest.mark.parametrize("change", [{"claimed_amount_minor": -1}, {"affected_ids": []}, {"type": "unknown_type"},
                                    {"confidence": 0.9}])
def test_wrong_but_well_formed_findings_are_false_positives_not_crashes(truth, change):
    """Определения жюри: структурно верный, но неправильный вывод — FP, а не неверная структура всего submit.
    Лишние ключи в выводе не делают структуру неверной."""
    env = AuditEnv()
    _, info = env.reset(3, FULL)
    issue = truth["issues"][0]
    env.step(submit(info["episode_id"], [finding_from_issue(issue, **change)]))
    report = env.final_report()
    assert report["status"] == "graded"
    expected_tp = 1 if "confidence" in change else 0
    assert (report["tp"], report["fp"]) == (expected_tp, 1 - expected_tp)


def test_limits_are_inclusive(truth):
    env = AuditEnv()
    _, info = env.reset(3, FULL)
    base = finding_from_issue(truth["issues"][0])
    ok = [{**base, "evidence_ids": base["evidence_ids"] + [f"X{i}" for i in range(10 - len(base["evidence_ids"]))],
           "explanation": "y" * 1000}] * 30
    assert len(validate_submission({"episode_id": info["episode_id"], "findings": ok}, info["episode_id"])) == 30


def test_submit_without_findings_wrapper_is_invalid():
    env = AuditEnv()
    env.reset(3, FULL)
    _, reward, terminated, _, _ = env.step({"tool": "submit", "args": {}})
    assert terminated and reward == INVALID_REWARD
    env.reset(3, FULL)
    _, reward, terminated, _, _ = env.step({"tool": "submit", "args": {"findings": {"episode_id": "x", "findings": []}, "extra": 1}})
    assert terminated and reward == INVALID_REWARD


def test_resubmission_after_the_end_is_rejected():
    env = AuditEnv()
    _, info = env.reset(3, FULL)
    env.step(submit(info["episode_id"], []))
    with pytest.raises(RuntimeError):
        env.step(submit(info["episode_id"], []))


def test_validate_submission_raises_structure_error():
    with pytest.raises(StructureError):
        validate_submission({"episode_id": "e", "findings": [1]}, "e")


def test_timeout_and_step_limit_score_an_empty_answer():
    env = AuditEnv()
    env.reset(3, CLEAN)
    for _ in range(100):
        env.step(calc())
    report = env.final_report()
    assert report["truncated"] and report["reward"] == pytest.approx(0.8)  # пустой ответ на чистой компании, 100 шагов
    env.reset(3, FULL)
    for _ in range(100):
        env.step(calc())
    assert env.final_report()["reward"] == -1.0
