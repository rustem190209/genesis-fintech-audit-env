"""Генерирует 10 train и 10 dev эпизодов в папку episodes/ (открытые файлы + эталон в evaluator/)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_env.files import write_episode  # noqa: E402
from audit_env.generator import generate  # noqa: E402
from audit_env.grader.validate import check_episode  # noqa: E402
from audit_env.plan import all_episodes  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="episodes", help="папка результата (по умолчанию episodes)")
    parser.add_argument("--no-closed", action="store_true", help="не писать truth.json и provenance.json")
    args = parser.parse_args(argv)
    out = Path(args.out)
    index = []
    for split, number, item in all_episodes():
        ep = generate(item["seed"], item["config"])
        problems = check_episode(ep)
        if problems:
            print(f"ОШИБКА в {split}/{item['name']}: {problems[:3]}")
            return 1
        folder = out / split / f"ep_{number:02d}_{item['name']}"
        write_episode(ep, folder, with_closed=not args.no_closed)
        index.append({"split": split, "dir": folder.relative_to(out).as_posix(), "seed": item["seed"],
                      "config": item["config"], "episode_id": ep.episode_id, "issues": len(ep.truth["issues"]),
                      "ledger_rows": len(ep.ledger), "documents": len(ep.invoices) + len(ep.payments)})
        print(f"{split}/{folder.name}: issues={len(ep.truth['issues'])} rows={len(ep.ledger)} docs={len(ep.invoices) + len(ep.payments)}")
    # newline="\n": файл должен быть одинаковым байт в байт на Windows и Linux
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"Готово: {len(index)} эпизодов в {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
