"""Прогоняет 4 базовые политики на открытых эпизодах: trajectories.jsonl и таблица метрик."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_env.plan import all_episodes  # noqa: E402
from audit_env.policies import POLICY_NAMES  # noqa: E402
from audit_env.runner import run_episode  # noqa: E402


def run_all(splits=("train", "dev"), trajectories=None):
    """Возвращает отчёты по всем (политика, эпизод). trajectories — открытый файл для записи шагов."""
    reports = []
    for split, number, item in all_episodes():
        if split not in splits:
            continue
        for name in POLICY_NAMES:
            steps = []
            report = run_episode(name, item["seed"], item["config"], mode="open", trajectory=steps)
            report.update({"split": split, "episode": f"{split}/{number:02d}_{item['name']}"})
            reports.append(report)
            if trajectories is not None:
                for line in steps:
                    trajectories.write(json.dumps({**line, "split": split}, ensure_ascii=False, sort_keys=True) + "\n")
    return reports


def summary(reports):
    """Средние метрики по политикам (отдельно: P, R, F1, награда, шаги, время)."""
    rows = ["| Политика | Эпизодов | P | R | F1 | Награда | Шаги | Время, с |", "|---|---|---|---|---|---|---|---|"]
    for name in POLICY_NAMES:
        mine = [r for r in reports if r["policy"] == name]
        if not mine:
            continue
        avg = lambda key: sum(r[key] for r in mine) / len(mine)
        rows.append(f"| {name} | {len(mine)} | {avg('precision'):.2f} | {avg('recall'):.2f} | {avg('f1'):.2f} | "
                    f"{avg('reward'):+.3f} | {avg('steps'):.1f} | {avg('wall_s'):.2f} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results", help="папка результата (по умолчанию results)")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "trajectories.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        reports = run_all(trajectories=handle)
    (out / "baseline_results.json").write_text(json.dumps(reports, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    table = summary(reports)
    (out / "baseline_results.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
