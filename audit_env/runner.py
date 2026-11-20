"""Прогон политики через среду с записью траектории (episode_id, step, action, observation, reward, ...)."""
import time

from .env import AuditEnv
from .policies import make_policy


def run_episode(policy_name: str, seed: int, config: dict | None = None, mode: str = "open",
                policy_seed: int = 0, trajectory: list | None = None) -> dict:
    """Один эпизод. Возвращает отчёт оценщика + время. trajectory (список) пополняется строками шагов."""
    env = AuditEnv(mode=mode)
    started = time.perf_counter()
    obs, info = env.reset(seed, config)
    truth = env.evaluator_truth() if policy_name == "oracle" else None  # только oracle получает эталон
    policy = make_policy(policy_name, policy_seed, truth)
    policy.start(obs, info)
    episode_id = info["episode_id"]
    if trajectory is not None:
        trajectory.append({"episode_id": episode_id, "policy": policy_name, "step": 0, "action": None,
                           "observation": obs, "reward": 0.0, "terminated": False, "truncated": False})
    done = False
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if trajectory is not None:
            trajectory.append({"episode_id": episode_id, "policy": policy_name, "step": info["step"], "action": action,
                               "observation": obs, "reward": reward, "terminated": terminated, "truncated": truncated})
    report = env.final_report()
    report["policy"] = policy_name
    report["wall_s"] = round(time.perf_counter() - started, 3)
    return report
