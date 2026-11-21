"""Тесты один в один по опубликованным «Определениям проверок FinTech» (20.11.2026).

Блок 1 — контракт C01–C20 через judge_adapter; блок 3 — 20 тестов matcher и награды; примеры M01–M12;
блок 4 — парные проверки: перестановка записей и смена ID в копии эпизода, подача через episode_dir.
"""
import json
import shutil
from fractions import Fraction

import pytest

import judge_adapter as ja
from audit_env.files import load_episode, load_open, load_truth, write_episode
from audit_env.generator import generate
from audit_env.grader import StructureError, match_findings, terminal_reward, validate_submission
from audit_env.grader.matcher import evidence_universe
from audit_env.grader.validate import check_episode, detect
from audit_env.transform import map_findings, transform_episode
from helpers import CLEAN, FULL, dump, oracle_findings

# =====================================================================================  блок 3

EP = "ep-fixture"
T, OTHER_T = "period_cutoff", "duplicate_invoice"
UNIVERSE = {"R1", "R2", "R3", "R4", "R5", "D1", "D2", "D3"}
A = {"issue_id": "A", "type": T, "affected_ids": ["R1"], "expected_amount_minor": 10000, "currency": "KZT",
     "tolerance_minor": 0, "evidence_sets": [["D1", "D2"]]}
B = {**A, "issue_id": "B", "affected_ids": ["R2"], "evidence_sets": [["D1", "D3"]]}


def F(**change):
    finding = {"type": T, "affected_ids": ["R1"], "claimed_amount_minor": 10000, "currency": "KZT",
               "evidence_ids": ["D1", "D2"], "explanation": "short reason"}
    finding.update(change)
    return finding


def score(findings, issues, steps=10, universe=UNIVERSE):
    """(MatchResult | None, награда). None — неверная структура всего submit."""
    try:
        parsed = validate_submission({"episode_id": EP, "findings": findings}, EP)
    except StructureError:
        return None, -1.0
    result = match_findings(parsed, issues, universe)
    return result, terminal_reward(result.f1, steps)


MASS = [F(type=t, affected_ids=[r]) for r in ("R1", "R2", "R3", "R4", "R5")
        for t in ("period_cutoff", "duplicate_invoice", "payment_allocation", "fx_error")]

BLOCK3 = [
    # № , findings, эталон, inspection_steps, (TP, FP, FN) или None, награда
    (1, [F()], [A], 10, (1, 0, 0), 0.98),
    (2, [], [A], 10, (0, 0, 1), -1.0),
    (3, [], [], 10, (0, 0, 0), 0.98),
    (4, [F()], [], 10, (0, 1, 0), -1.0),
    (5, [F(), F()], [A], 10, (1, 1, 0), float(Fraction(47, 150))),
    (6, [F(type=OTHER_T)], [A], 10, (0, 1, 1), -1.0),
    (7, [F(affected_ids=["R2"])], [A], 10, (0, 1, 1), -1.0),
    (8, [F(claimed_amount_minor=10001)], [A], 10, (0, 1, 1), -1.0),
    (9, [F(currency="USD")], [A], 10, (0, 1, 1), -1.0),
    (10, [F(evidence_ids=["D1"])], [A], 10, (0, 1, 1), -1.0),
    (11, [F(evidence_ids=["D1", "U-absent"])], [A], 10, (0, 1, 1), -1.0),
    (12, MASS, [A], 10, (1, 19, 0), float(Fraction(-871, 1050))),
    (13, [F()], [A, B], 10, (1, 0, 1), float(Fraction(47, 150))),
    (14, [F(), F(affected_ids=["R3"])], [A, B], 10, (1, 1, 1), -0.02),
    (15, [F()], [A], 0, (1, 0, 0), 1.0),
    (16, [F()], [A], 50, (1, 0, 0), 0.90),
    (17, [F()], [A], 100, (1, 0, 0), 0.80),
    (18, [F()] * 31, [A], 10, None, -1.0),
    (19, [F(evidence_ids=["D1", "D2", "D3", "R1", "R2", "R3", "R4", "R5", "X1", "X2", "X3"])], [A], 10, None, -1.0),
    (20, [F(explanation="a" * 1001)], [A], 10, None, -1.0),
]


@pytest.mark.parametrize("number,findings,issues,steps,counts,expected", BLOCK3, ids=[f"test{c[0]}" for c in BLOCK3])
def test_block3_matcher_and_reward(number, findings, issues, steps, counts, expected):
    result, reward = score(findings, issues, steps)
    assert reward == pytest.approx(expected, abs=1e-9)
    if counts is None:
        assert result is None  # неверная структура
    else:
        assert (result.tp, result.fp, result.fn) == counts


def test_block3_metrics_values():
    r, _ = score([F()], [A])
    assert (r.precision, r.recall, r.f1) == (1, 1, 1)
    r, _ = score([], [])
    assert (r.precision, r.recall, r.f1) == (0, 0, 1)
    r, _ = score([F(), F()], [A])
    assert (r.precision, r.recall) == (0.5, 1) and r.f1 == pytest.approx(2 / 3)
    r, _ = score(MASS, [A])
    assert r.precision == pytest.approx(1 / 20) and r.f1 == pytest.approx(2 / 21)
    r, _ = score([F(), F(affected_ids=["R3"])], [A, B])
    assert (r.precision, r.recall, r.f1) == (0.5, 0.5, 0.5)
    assert score([F()], [A], 50)[1] - score([F()], [A], 10)[1] == pytest.approx(-0.08)


# =====================================================================================  M01–M12

A2 = {**A, "affected_ids": ["R1", "R2"]}
B2 = {**A2, "issue_id": "B", "evidence_sets": [["D3"]]}


def G(**change):
    change.setdefault("affected_ids", ["R1", "R2"])
    return F(**change)


@pytest.mark.parametrize("name,findings,counts", [
    ("M01", [G(affected_ids=["R2", "R1", "R1"])], (1, 0, 0)),
    ("M02", [G(affected_ids=["R1", "R2", "R3"])], (0, 1, 1)),
    ("M03", [G(evidence_ids=["D2", "D1", "D3"])], (1, 0, 0)),
    ("M04", [G(evidence_ids=["D1", "D2", "U"])], (0, 1, 1)),
    ("M06a", [G(currency="kzt")], (0, 1, 1)),
    ("M06b", [G(currency="KZT ")], (0, 1, 1)),
    ("M07", [G(claimed_amount_minor=10001)], (0, 1, 1)),
    ("M12", [G(), G()], (1, 1, 0)),
])
def test_open_examples(name, findings, counts):
    result, _ = score(findings, [A2])
    assert (result.tp, result.fp, result.fn) == counts


def test_m05_eleven_evidence_with_repeats_is_a_structure_error():
    assert score([G(evidence_ids=["D1"] * 10 + ["D2"])], [A2]) == (None, -1.0)


def test_m08_published_tolerance_is_inclusive():
    issue = {**A2, "tolerance_minor": 1}
    assert score([G(claimed_amount_minor=10001)], [issue])[0].tp == 1
    assert score([G(claimed_amount_minor=10002)], [issue])[0].tp == 0


@pytest.mark.parametrize("amount", ["10000", True, 10000.0])
def test_m09_amount_must_be_a_json_integer(amount):
    assert score([G(claimed_amount_minor=amount)], [A2]) == (None, -1.0)


def test_m10_maximum_matching_not_greedy():
    f0, f1 = G(evidence_ids=["D1", "D2", "D3"]), G(evidence_ids=["D1", "D2"])  # F0 подходит A и B, F1 — только A
    result, _ = score([f0, f1], [A2, B2])
    assert (result.tp, result.fp, result.fn) == (2, 0, 0)
    assert sorted(result.pairs) == [("A", 1), ("B", 0)]


def test_m11_deterministic_choice_and_order_independence():
    f = G(evidence_ids=["D1", "D2", "D3"])
    result, reward = score([f, dict(f)], [A2, B2])
    assert result.tp == 2 and result.pairs == [("A", 0), ("B", 1)]
    swapped, reward2 = score([dict(f), f], [B2, A2])
    assert swapped.tp == 2 and reward2 == reward


# =====================================================================================  блок 1 (адаптер)

U = "ZZZ-NOT-IN-FIXTURE"


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp("judge")
    paths = {}
    for name, seed, config in (("C", 11, CLEAN), ("E", 12, {"errors": {"period_cutoff": 1}, "n_journal": 150}),
                               ("L", 13, FULL)):
        paths[name] = root / name
        ja.generate_episode(seed, config, str(paths[name]))
    paths["open_only"] = root / "open_only"  # копия без закрытых файлов: среда должна работать и так
    write_episode(load_episode(paths["C"], with_closed=False), paths["open_only"])
    return paths


def closed(path):
    return {"mode": "closed", "episode_dir": str(path), "team_config": {}}


def act(name, **args):
    return {"name": name, "args": args}


def fresh(fixtures, name="C"):
    env = ja.make_env()
    obs, info = env.reset(0, closed(fixtures[name]))
    return env, obs, info


def test_c01_reset_returns_json_pair(fixtures):
    env, obs, info = fresh(fixtures)
    json.dumps([obs, info])


def test_c02_first_observation(fixtures):
    env, obs, info = fresh(fixtures)
    assert set(obs) == {"task", "manifest", "policy", "remaining_steps"} and obs["remaining_steps"] == 100
    assert "seed" not in dump(obs) and "evidence_sets" not in dump(obs)


def test_c03_manifest_numbers_match_the_data(fixtures):
    env, obs, info = fresh(fixtures)
    m, data = obs["manifest"], load_open(fixtures["C"])
    for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end", "timezone", "rounding"):
        assert key in m
    assert m["counts"]["ledger_rows"] == len(data["ledger"])
    assert m["counts"]["documents"] == len(data["invoices"]) + len(data["payments"])


def test_c04_step_types(fixtures):
    env, _, _ = fresh(fixtures)
    obs, reward, terminated, truncated, info = env.step(act("list_records", table="ledger", cursor=None, limit=5))
    assert isinstance(terminated, bool) and isinstance(truncated, bool) and isinstance(reward, float)
    assert set(obs) == {"result", "error", "remaining_steps"} and set(obs["result"]) == {"records", "next_cursor"}
    json.dumps([obs, reward, info])


def test_c05_c06_pages_and_stable_cursors(fixtures):
    env, _, _ = fresh(fixtures, "L")
    rows, cursor, pages = [], None, 0
    while True:
        res = env.step(act("list_records", table="ledger", cursor=cursor, limit=50))[0]["result"]
        assert len(res["records"]) <= 50
        rows += res["records"]
        pages += 1
        cursor = res["next_cursor"]
        if cursor is None:
            break
    assert pages >= 3 and [r["row_id"] for r in rows] == [r["row_id"] for r in load_open(fixtures["L"])["ledger"]]
    first = env.step(act("list_records", table="ledger", cursor="50", limit=50))[0]
    again = env.step(act("list_records", table="ledger", cursor="50", limit=50))[0]
    assert first["result"] == again["result"] and again["remaining_steps"] == first["remaining_steps"] - 1


def test_c07_c08_record_and_document(fixtures):
    data = load_open(fixtures["C"])
    env, _, _ = fresh(fixtures)
    r1, d1 = data["ledger"][0], data["invoices"][0]
    assert env.step(act("get_record", table="ledger", id=r1["row_id"]))[0]["result"] == r1
    assert env.step(act("get_document", doc_id=d1["doc_id"]))[0]["result"] == d1


def test_c09_c10_calculator(fixtures):
    env, _, _ = fresh(fixtures)
    assert env.step(act("calculator", expression="0.1+0.2"))[0]["result"] == "0.3"
    obs = env.step(act("calculator", expression="open('/etc/passwd').read()"))[0]
    assert obs["result"] is None and obs["error"]["code"] == "calc_error"


@pytest.mark.parametrize("action", [act("get_record", table="ledger", id=U), act("get_document", doc_id=U),
                                    act("teleport", where="x")])
def test_c11_c12_c13_bounded_errors(fixtures, action):
    env, _, _ = fresh(fixtures)
    obs = env.step(action)[0]
    assert obs["error"] and obs["result"] is None and obs["remaining_steps"] == 99
    assert set(obs["error"]) == {"code", "message"}


def test_c14_errors_cost_a_step_and_do_not_break_access(fixtures):
    data = load_open(fixtures["C"])
    env, _, _ = fresh(fixtures)
    a = env.step(act("get_record", table="ledger", id=data["ledger"][1]["row_id"]))[0]
    b = env.step(act("get_record", table="ledger", id=U))[0]
    c = env.step(act("get_record", table="ledger", id=data["ledger"][1]["row_id"]))[0]
    assert (a["remaining_steps"], b["remaining_steps"], c["remaining_steps"]) == (99, 98, 97) and a["result"] == c["result"]


def test_c15_submit_is_final(fixtures):
    env, obs, info = fresh(fixtures)
    payload = {"episode_id": info["episode_id"], "findings": []}
    obs, reward, terminated, truncated, _ = env.step(act("submit", findings=payload))
    assert terminated and not truncated and obs["result"] == {"accepted": True}
    after = env.step(act("submit", findings={"episode_id": info["episode_id"], "findings": [F()]}))
    assert after[2] is True and after[0]["error"]["code"] == "episode_over"
    assert env._env.final_report()["submission"] == payload  # зафиксирован первый ответ


def test_c16_submit_as_action_100(fixtures):
    env, obs, info = fresh(fixtures)
    for _ in range(99):
        env.step(act("calculator", expression="1+1"))
    obs, _, terminated, truncated, _ = env.step(act("submit", findings={"episode_id": info["episode_id"], "findings": []}))
    assert terminated and not truncated and obs["result"] == {"accepted": True}


def test_c17_hundred_actions_without_submit(fixtures):
    env, _, _ = fresh(fixtures)
    for _ in range(100):
        obs, _, terminated, truncated, _ = env.step(act("calculator", expression="1+1"))
    assert truncated and not terminated
    later = env.step(act("calculator", expression="1+1"))
    assert later[0]["error"]["code"] == "episode_over" and later[3] is True
    assert env._env.final_report()["submission"] is None  # итоговые findings пусты


def test_c18_time_limit_includes_policy_time(fixtures):
    class Clock:
        t = 0.0

        def __call__(self):
            return self.t

    clock = Clock()
    env = ja.JudgeEnv(clock=clock)
    _, info = env.reset(0, closed(fixtures["C"]))
    env.step(act("calculator", expression="1+1"))
    clock.t = 61.0  # политика думала дольше лимита
    obs, _, terminated, truncated, _ = env.step(act("submit", findings={"episode_id": info["episode_id"], "findings": []}))
    assert truncated and not terminated and obs["error"]["code"] == "time_limit"
    assert env._env.final_report()["submission"] is None


def test_c19_closed_mode_rewards_are_zero(fixtures):
    env, obs, info = fresh(fixtures, "E")
    truth = load_truth(fixtures["E"])
    rewards = [env.step(act("list_records", table="invoices"))[1] for _ in range(3)]
    obs, reward, terminated, _, info2 = env.step(act("submit", findings={"episode_id": info["episode_id"],
                                                                         "findings": oracle_findings(truth)}))
    assert rewards == [0.0, 0.0, 0.0] and reward == 0.0 and terminated
    assert set(info2) == {"episode_id", "step"} and obs["result"] == {"accepted": True}
    graded = ja.score_episode(str(fixtures["E"]), {"episode_id": info["episode_id"], "findings": oracle_findings(truth)}, 3)
    assert graded["return"] == pytest.approx(1 - 0.002 * 3)  # истинная награда есть только у оценщика


def test_c19_train_mode_returns_the_terminal_reward(fixtures):
    env = ja.make_env()
    _, info = env.reset(0, {"mode": "train", "episode_dir": str(fixtures["E"]), "team_config": {}})
    truth = load_truth(fixtures["E"])
    reward = env.step(act("submit", findings={"episode_id": info["episode_id"], "findings": oracle_findings(truth)}))[1]
    assert reward == 1.0


def test_c20_no_leaks_through_any_channel(fixtures):
    outputs = []
    for name in ("C", "E"):
        data = load_open(fixtures[name])
        ids = [d["doc_id"] for d in data["invoices"] + data["payments"]]
        for start in range(0, len(ids), 90):  # повторный reset начинает эпизод заново
            env, obs, info = fresh(fixtures, name)
            outputs.append([obs, info])
            for doc in ids[start:start + 90]:
                outputs.append(env.step(act("get_document", doc_id=doc)))
        env, _, _ = fresh(fixtures, name)
        for table in ("ledger", "invoices", "payments", "fx"):
            cursor = None
            while True:
                out = env.step(act("list_records", table=table, cursor=cursor, limit=50))
                outputs.append(out)
                cursor = out[0]["result"]["next_cursor"]
                if cursor is None:
                    break
        probe = env.step(act("get_document", doc_id="../../manifest.json"))  # попытка выйти в файловую систему
        assert probe[0]["result"] is None and probe[0]["error"]["code"] == "not_found"
    text = dump(outputs)
    for word in ("ISS-", "truth", "provenance", "evidence_sets", "expected_amount", "clean_ledger", "seed",
                 "evaluator", "tolerance_minor"):
        assert word not in text, word


def test_closed_mode_never_needs_the_evaluator_files(fixtures):
    env = ja.make_env()
    obs, info = env.reset(0, closed(fixtures["open_only"]))
    out = env.step(act("submit", findings={"episode_id": info["episode_id"], "findings": []}))
    assert out[2] and out[0]["result"] == {"accepted": True}
    with pytest.raises(FileNotFoundError):
        load_truth(fixtures["open_only"])


def test_submit_structure_error_is_bounded_and_scored_minus_one(fixtures):
    env, _, info = fresh(fixtures)
    obs, reward, terminated, _, _ = env.step(act("submit", findings={"episode_id": info["episode_id"], "findings": "x"}))
    assert terminated and reward == 0.0 and obs["error"]["code"] == "invalid_submission"
    graded = ja.score_episode(str(fixtures["C"]), {"episode_id": info["episode_id"], "findings": "x"}, 0)
    assert graded == {"tp": None, "fp": None, "fn": None, "precision": None, "recall": None, "f1": None, "return": -1.0}


def test_score_episode_output_shape(fixtures):
    truth = load_truth(fixtures["E"])
    graded = ja.score_episode(str(fixtures["E"]), {"episode_id": truth["episode_id"], "findings": oracle_findings(truth)}, 10)
    assert set(graded) == {"tp", "fp", "fn", "precision", "recall", "f1", "return"}
    assert (graded["tp"], graded["fp"], graded["fn"]) == (1, 0, 0) and graded["return"] == pytest.approx(0.98)


def test_generate_episode_accepts_team_config_wrapper(tmp_path):
    ja.generate_episode(5, {"team_config": FULL}, str(tmp_path / "a"))
    ja.generate_episode(5, FULL, str(tmp_path / "b"))
    assert (tmp_path / "a" / "ledger.csv").read_bytes() == (tmp_path / "b" / "ledger.csv").read_bytes()
    assert (tmp_path / "a" / "evaluator" / "truth.json").is_file()


def test_train_mode_without_episode_dir_uses_the_generator():
    env = ja.make_env()
    obs, info = env.reset(7, {"mode": "train", "team_config": FULL})
    assert info["episode_id"] == generate(7, FULL).episode_id


# =====================================================================================  блок 4

KINDS = {"permute": (True, False), "remap": (False, True), "both": (True, True)}


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("seed", [21, 22, 23])
def test_block4_paired_checks(tmp_path, seed, kind):
    src, dst = tmp_path / "src", tmp_path / "dst"
    config = FULL if seed != 21 else CLEAN
    ja.generate_episode(seed, config, str(src))
    permute, remap = KINDS[kind]
    maps = transform_episode(src, dst, permute=permute, remap=remap, seed=seed)
    truth_a, truth_b = load_truth(src), load_truth(dst)
    answer = {"episode_id": truth_a["episode_id"], "findings": oracle_findings(truth_a)}
    moved = map_findings(answer, maps)
    # 2-3: oracle с новыми ID даёт те же метрики и награду при одинаковых inspection_steps
    assert ja.score_episode(str(src), answer, 7) == ja.score_episode(str(dst), moved, 7)
    assert ja.score_episode(str(dst), moved, 7)["return"] == pytest.approx(1 - 0.014) or not truth_a["issues"]
    # обратное отображение ID: тот же эталон
    back = {new: old for space in maps.values() for old, new in space.items()}
    restore = lambda issue: {**issue, "affected_ids": sorted(back.get(x, x) for x in issue["affected_ids"]),
                             "evidence_sets": [sorted(back.get(x, x) for x in s) for s in issue["evidence_sets"]]}
    assert [restore(i) for i in truth_b["issues"]] == truth_a["issues"]
    # 4: данные по-прежнему корректны, доказательства доступны, новых каналов утечки нет
    assert check_episode(load_episode(dst, with_closed=True)) == []
    found = validate_submission({"episode_id": truth_b["episode_id"], "findings": detect(load_open(dst))},
                                truth_b["episode_id"])
    result = match_findings(found, truth_b["issues"], evidence_universe(load_open(dst)))
    assert (result.fp, result.fn) == (0, 0)
    # среда принимает преобразованную копию через episode_dir
    env = ja.make_env()
    obs, info = env.reset(0, closed(dst))
    assert info["episode_id"] == truth_b["episode_id"]


def test_block4_original_replay_is_exact(tmp_path):
    ja.generate_episode(31, FULL, str(tmp_path / "ep"))

    def play():
        env = ja.make_env()
        out = [env.reset(0, closed(tmp_path / "ep"))]
        for action in (act("list_records", table="ledger", limit=7), act("list_records", table="fx", cursor="3"),
                       act("get_record", table="ledger", id=U), act("calculator", expression="2/3")):
            out.append(env.step(action))
        return dump(out)

    assert play() == play()


def test_transform_keeps_line_item_order_and_fx_keys(tmp_path):
    ja.generate_episode(32, FULL, str(tmp_path / "src"))
    transform_episode(tmp_path / "src", tmp_path / "dst", permute=True, remap=True, seed=1)
    a, b = load_open(tmp_path / "src"), load_open(tmp_path / "dst")
    lines = lambda data: sorted(json.dumps(i["line_items"]) for i in data["invoices"])
    assert lines(a) == lines(b)
    assert sorted((r["date"], r["source_currency"]) for r in a["fx"]) == sorted((r["date"], r["source_currency"]) for r in b["fx"])
    assert {r["row_id"] for r in a["ledger"]}.isdisjoint({r["row_id"] for r in b["ledger"]})
