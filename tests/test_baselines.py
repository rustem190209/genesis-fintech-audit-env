"""Базовые политики: порядок наград, траектории, воспроизводимость."""
import json

import pytest

from audit_env.policies import POLICY_NAMES
from audit_env.runner import run_episode
from helpers import CLEAN, FULL, dump

SEEDS = (1, 2, 3, 4)


def mean_reward(policy, config):
    return sum(run_episode(policy, s, config)["reward"] for s in SEEDS) / len(SEEDS)


def test_policy_ranking_on_faulty_books():
    oracle, recon = mean_reward("oracle", FULL), mean_reward("simple_reconciler", FULL)
    rand, flag = mean_reward("random_explorer", FULL), mean_reward("flag_all", FULL)
    assert oracle == 1.0
    assert oracle > recon > max(rand, flag)
    assert rand <= -0.9 and flag <= -0.9


def test_policy_ranking_on_clean_books():
    assert mean_reward("oracle", CLEAN) == 1.0
    assert mean_reward("flag_all", CLEAN) == -1.0 and mean_reward("random_explorer", CLEAN) <= 0


def test_oracle_uses_no_inspection_steps():
    report = run_episode("oracle", 1, FULL)
    assert report["steps"] == 1 and report["inspection_steps"] == 0 and report["reward"] == 1.0


def test_trajectory_format_and_no_leaks():
    for name in POLICY_NAMES:
        trajectory = []
        run_episode(name, 2, FULL, trajectory=trajectory)
        assert trajectory[0]["step"] == 0 and trajectory[-1]["terminated"] is True
        for line in trajectory:
            assert {"episode_id", "step", "action", "observation", "reward", "terminated", "truncated"} <= set(line)
        text = dump([t["observation"] for t in trajectory])
        for word in ("ISS-", "evidence_sets", "expected_amount", "provenance", "clean_ledger"):
            assert word not in text
        assert [t["step"] for t in trajectory] == list(range(len(trajectory)))
        json.dumps(trajectory)  # траектория целиком JSON-совместима


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_policies_are_reproducible(name):
    a, b = [], []
    ra = run_episode(name, 3, FULL, policy_seed=5, trajectory=a)
    rb = run_episode(name, 3, FULL, policy_seed=5, trajectory=b)
    assert a == b and ra["reward"] == rb["reward"]


def test_random_explorer_depends_on_its_seed():
    a, b = [], []
    run_episode("random_explorer", 3, FULL, policy_seed=1, trajectory=a)
    run_episode("random_explorer", 3, FULL, policy_seed=2, trajectory=b)
    assert [t["action"] for t in a] != [t["action"] for t in b]


def test_runs_fit_the_offline_limits():
    for name in POLICY_NAMES:
        report = run_episode(name, 1, {**FULL, "n_journal": 700})
        assert report["steps"] <= 100 and report["wall_s"] < 60
