"""Безвредные преобразования эпизода, как в парных проверках жюри (блок 4 определений проверок).

Жюри берёт сгенерированный эпизод, делает копию, переставляет записи и/или согласованно меняет ID
во всех данных и ссылках, правит закрытые truth и provenance и подаёт копию в reset через episode_dir.
Здесь то же самое — чтобы заранее убедиться, что наша среда и оценщик это выдерживают.

Правила (как в определениях):
* переставляются строки таблиц и потоки документов, но не line_items внутри счёта;
* ID меняются взаимно однозначно в каждом пространстве отдельно: row_id, doc_id, invoice_id,
  payment_id, entity_id — везде, где на них ссылаются (в том числе affected_ids и evidence_ids);
* id курса 'ДАТА:SRC:DST' не меняется (это не суррогатный ID);
* суммы, null и связи сохраняются.
"""
import copy
import random

from .files import load_open, load_provenance, load_truth, write_episode
from .generator import Episode

NAMESPACES = ("row_id", "doc_id", "invoice_id", "payment_id", "entity_id")


def _new_ids(rng, ids, prefix):
    numbers = rng.sample(range(10 ** 6, 10 ** 7), len(ids))
    return {old: f"{prefix}{num}" for old, num in zip(sorted(ids), numbers)}


def build_id_maps(data: dict, seed: int) -> dict:
    """Отображения старый ID -> новый для каждого пространства ID."""
    rng = random.Random(seed)
    found = {
        "row_id": {r["row_id"] for r in data["ledger"]},
        "doc_id": {d["doc_id"] for d in data["invoices"] + data["payments"]},
        "invoice_id": {i["invoice_id"] for i in data["invoices"]},
        "payment_id": {p["payment_id"] for p in data["payments"]},
        "entity_id": {i["entity_id"] for i in data["invoices"]} | {r["entity_id"] for r in data["ledger"] if r["entity_id"]},
    }
    prefixes = {"row_id": "R", "doc_id": "D", "invoice_id": "I", "payment_id": "P", "entity_id": "E"}
    return {ns: _new_ids(rng, sorted(found[ns]), prefixes[ns] + "x") for ns in NAMESPACES}


def transform_episode(src, dst, permute: bool = True, remap: bool = True, seed: int = 0) -> dict:
    """Копия эпизода src в dst с перестановкой и/или сменой ID. Возвращает отображения ID (для обратной сверки)."""
    data = load_open(src)
    truth, provenance = load_truth(src), load_provenance(src)
    maps = build_id_maps(data, seed) if remap else {ns: {} for ns in NAMESPACES}
    m = lambda ns, value: None if value is None else maps[ns].get(value, value)
    any_id = {}
    for ns in ("row_id", "doc_id", "invoice_id", "payment_id"):
        for old, new in maps[ns].items():
            assert old not in any_id, "ID spaces overlap; evidence mapping would be ambiguous"
            any_id[old] = new

    ledger = [{**r, "row_id": m("row_id", r["row_id"]), "entity_id": m("entity_id", r["entity_id"]),
               "invoice_id": m("invoice_id", r["invoice_id"]), "payment_id": m("payment_id", r["payment_id"])}
              for r in data["ledger"]]
    invoices = [{**i, "doc_id": m("doc_id", i["doc_id"]), "invoice_id": m("invoice_id", i["invoice_id"]),
                 "entity_id": m("entity_id", i["entity_id"]), "line_items": copy.deepcopy(i["line_items"])}
                for i in data["invoices"]]
    payments = [{**p, "doc_id": m("doc_id", p["doc_id"]), "payment_id": m("payment_id", p["payment_id"]),
                 "invoice_refs": [{**ref, "invoice_id": m("invoice_id", ref["invoice_id"])} for ref in p["invoice_refs"]]}
                for p in data["payments"]]
    fx = [dict(r) for r in data["fx"]]
    if permute:
        rng = random.Random(seed + 1)
        for table in (ledger, invoices, payments, fx):
            rng.shuffle(table)

    issues = []
    for issue in truth["issues"]:
        space = "payment_id" if issue["type"] in ("payment_allocation", "fx_error") else "invoice_id"
        issues.append({**issue, "affected_ids": sorted(m(space, x) for x in issue["affected_ids"]),
                       "evidence_sets": [sorted(any_id.get(x, x) for x in s) for s in issue["evidence_sets"]]})
    new_truth = {**truth, "issues": issues}
    new_prov = copy.deepcopy(provenance)
    new_prov["clean_ledger"] = [{**r, "entity_id": m("entity_id", r["entity_id"]),
                                 "invoice_id": m("invoice_id", r["invoice_id"]),
                                 "payment_id": m("payment_id", r["payment_id"])} for r in provenance["clean_ledger"]]
    for change in new_prov["changes"]:
        if change.get("payment_id"):
            change["payment_id"] = m("payment_id", change["payment_id"])
    new_prov["transform"] = {"permute": permute, "remap": remap, "seed": seed}

    episode = Episode(episode_id=data["manifest"]["episode_id"], manifest=data["manifest"], policy=data["policy"],
                      ledger=ledger, invoices=invoices, payments=payments, fx=fx, truth=new_truth, provenance=new_prov)
    write_episode(episode, dst, with_closed=True)
    return maps


def map_findings(findings: dict, maps: dict) -> dict:
    """Те же выводы с новыми ID (так oracle подаёт ответ на преобразованном варианте)."""
    any_id = {old: new for ns in NAMESPACES for old, new in maps[ns].items()}
    return {**findings, "findings": [{**f, "affected_ids": [any_id.get(x, x) for x in f["affected_ids"]],
                                      "evidence_ids": [any_id.get(x, x) for x in f["evidence_ids"]]}
                                     for f in findings["findings"]]}
