"""Запись эпизода в папку и чтение открытых файлов обратно."""
import csv
import json
from pathlib import Path

from .generator import Episode


def write_episode(ep: Episode, folder, with_closed: bool = False) -> Path:
    """Открытые файлы кладём в folder. Закрытые (truth, provenance) только по явной просьбе
    и в подпапку evaluator/: это эталон открытых train/dev-эпизодов, не закрытых."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in ep.open_files().items():
        (folder / name).write_text(text, encoding="utf-8", newline="\n")
    if with_closed:
        closed = folder / "evaluator"
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
