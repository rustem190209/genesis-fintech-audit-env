"""Единая команда проверки: python check.py

1. тесты (pytest);
2. 20 открытых эпизодов: корректность данных и эталона, повторная генерация даёт те же байты;
3. четыре базовые политики и порядок их наград.
Без сети, платных API и GPU. Код возврата 0 = всё в порядке.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audit_env.generator import generate  # noqa: E402
from audit_env.grader.validate import check_episode  # noqa: E402
from audit_env.plan import all_episodes  # noqa: E402
from scripts.run_baselines import run_all, summary  # noqa: E402


def step(title):
    print(f"\n=== {title} ===", flush=True)


def main() -> int:
    failed = []

    step("1/3 Тесты")
    if subprocess.call([sys.executable, "-m", "pytest"], cwd=ROOT) != 0:
        failed.append("тесты")

    step("2/3 Эпизоды: данные, эталон, воспроизводимость")
    bad = 0
    for split, number, item in all_episodes():
        ep = generate(item["seed"], item["config"])
        problems = check_episode(ep)
        again = generate(item["seed"], item["config"])
        if ep.open_files() != again.open_files() or ep.closed_files() != again.closed_files():
            problems.append("повторная генерация дала другие данные")
        if problems:
            bad += 1
            print(f"  {split}/{number:02d} {item['name']}: {problems[:3]}")
    print(f"  проверено эпизодов: {len(all_episodes())}, с проблемами: {bad}")
    if bad:
        failed.append("эпизоды")

    step("3/3 Базовые политики")
    reports = run_all()
    print(summary(reports))
    mean = lambda name: sum(r["reward"] for r in reports if r["policy"] == name) / sum(1 for r in reports if r["policy"] == name)
    ok = mean("oracle") == 1.0 and mean("oracle") > mean("simple_reconciler") > max(mean("random_explorer"), mean("flag_all"))
    print("  порядок наград oracle > simple_reconciler > random/flag_all:", "да" if ok else "НЕТ")
    if not ok:
        failed.append("порядок наград")

    print()
    if failed:
        print("ПРОВЕРКА НЕ ПРОЙДЕНА:", ", ".join(failed))
        return 1
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
