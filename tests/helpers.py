"""Общие помощники для тестов."""
import json

CLEAN = {"n_journal": 150}
FULL = {"errors": {"duplicate_invoice": 1, "payment_allocation": 1, "fx_error": 1, "period_cutoff": 1},
        "interactions": ["fx_boundary_partial", "dup_cutoff"], "n_journal": 150}
TYPES = ("duplicate_invoice", "payment_allocation", "fx_error", "period_cutoff")


def single(error_type, n=1):
    return {"errors": {error_type: n}, "n_journal": 150}


def calc(expression="1+1"):
    return {"tool": "calculator", "args": {"expression": expression}}


def submit(episode_id, findings):
    return {"tool": "submit", "args": {"findings": {"episode_id": episode_id, "findings": findings}}}


def finding_from_issue(issue, **override):
    data = {"type": issue["type"], "affected_ids": list(issue["affected_ids"]),
            "claimed_amount_minor": issue["expected_amount_minor"], "currency": issue["currency"],
            "evidence_ids": list(issue["evidence_sets"][0]), "explanation": "test"}
    data.update(override)
    return data


def oracle_findings(truth):
    return [finding_from_issue(i) for i in truth["issues"]]


def dump(obj):
    return json.dumps(obj, sort_keys=True)
