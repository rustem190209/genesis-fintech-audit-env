"""Запись эпизода в папку и чтение обратно.

Раскладка папки эпизода:
  manifest.json, policy.json, ledger.csv, invoices.jsonl, payments.jsonl, fx.csv   — открытые файлы
  evaluator/truth.json, evaluator/provenance.json                                  — закрытые файлы
Среда в закрытом режиме читает только открытые файлы. Эталон читает оценщик (score_episode).
"""
import csv
import json
from pathlib import Path

from .generator import Episode

CLOSED_DIR = "evaluator"


def write_episode(ep: Episode, folder, with_closed: bool = False) -> Path:
    """Открытые файлы кладём в folder; закрытые (truth, provenance) — по просьбе, в подпапку evaluator/."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in ep.open_files().items():
        (folder / name).write_text(text, encoding="utf-8", newline="\n")
    if with_closed:
        closed = folder / CLOSED_DIR
        closed.mkdir(exist_ok=True)
        for name, text in ep.closed_files().items():
            (closed / name).write_text(text, encoding="utf-8", newline="\n")
    return folder


def load_open(folder) -> dict:
    """Читает открытые файлы эпизода в те же структуры, что видит среда."""
    folder = Path(folder)

    def cell(value):
        return None if value == "null" else value

    with open(folder / "ledger.csv", encoding="utf-8", newline="") as handle:
        ledger = []
        for row in csv.DictReader(handle):
            row = {k: cell(v) for k, v in row.items()}
            row["debit_minor"], row["credit_minor"] = int(row["debit_minor"]), int(row["credit_minor"])
            ledger.append(row)
    with open(folder / "fx.csv", encoding="utf-8", newline="") as handle:
        fx = list(csv.DictReader(handle))

    def jsonl(name):
        return [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines() if line]

    return {"manifest": json.loads((folder / "manifest.json").read_text(encoding="utf-8")),
            "policy": json.loads((folder / "policy.json").read_text(encoding="utf-8")),
            "ledger": ledger, "invoices": jsonl("invoices.jsonl"), "payments": jsonl("payments.jsonl"), "fx": fx}


def _closed_file(folder, name) -> Path:
    """Закрытый файл: evaluator/<name> (наша раскладка) или <name> в корне эпизода."""
    folder = Path(folder)
    for path in (folder / CLOSED_DIR / name, folder / name):
        if path.is_file():
            return path
    raise FileNotFoundError(f"{name} not found in the episode folder")


def load_truth(folder) -> dict:
    return json.loads(_closed_file(folder, "truth.json").read_text(encoding="utf-8"))


def load_provenance(folder) -> dict:
    return json.loads(_closed_file(folder, "provenance.json").read_text(encoding="utf-8"))


def load_episode(folder, with_closed: bool) -> Episode:
    """Эпизод из папки. with_closed=False — без эталона (так работает среда в закрытом режиме)."""
    data = load_open(folder)
    return Episode(episode_id=data["manifest"]["episode_id"], manifest=data["manifest"], policy=data["policy"],
                   ledger=data["ledger"], invoices=data["invoices"], payments=data["payments"], fx=data["fx"],
                   truth=load_truth(folder) if with_closed else None,
                   provenance=load_provenance(folder) if with_closed else None)
