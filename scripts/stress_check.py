r"""Стресс-проверка генератора: много seed x несколько config, на каждом эпизоде все проверки данных.

  python scripts\stress_check.py            (300 seed x 6 config = 1800 эпизодов, около минуты)
  python scripts\stress_check.py --seeds 50
Код возврата 0 = ни одного эпизода с проблемами.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audit_env.generator import generate  # noqa: E402
from audit_env.grader.validate import check_episode  # noqa: E402

ALL = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_error": 1, "period_cutoff": 1}
CONFIGS = [
    {"n_journal": 150},
    {"errors": {t: 3 for t in ALL}, "n_journal": 150},
    {"errors": ALL, "interactions": ["fx_boundary_partial", "dup_cutoff"], "n_journal": 400},
    {"errors": {"fx_error": 4}, "interactions": ["fx_boundary_partial"], "n_journal": 150, "n_invoices": 40},
    {"errors": {"period_cutoff": 6}, "n_journal": 150, "base_currency": "KZT"},
    {"errors": {"payment_allocation": 5, "fx_error": 5}, "n_journal": 150, "base_currency": "EUR", "n_invoices": 80},
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=300, help="сколько seed на каждый config (по умолчанию 300)")
    parser.add_argument("--first", type=int, default=1000, help="первый seed (по умолчанию 1000)")
    args = parser.parse_args(argv)
    total = bad = 0
    for seed in range(args.first, args.first + args.seeds):
        for number, config in enumerate(CONFIGS):
            total += 1
            try:
                problems = check_episode(generate(seed, config))
            except Exception as exc:  # любое падение генератора — тоже проблема
                problems = [f"{type(exc).__name__}: {exc}"]
            if problems:
                bad += 1
                if bad <= 10:
                    print(f"seed={seed} config#{number}: {problems[:2]}")
    print(f"эпизодов: {total}, с проблемами: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
