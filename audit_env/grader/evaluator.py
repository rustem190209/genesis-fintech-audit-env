"""Оценщик эпизода. Держит закрытый эталон; агент к нему доступа не имеет."""
from .matcher import StructureError, match_findings, validate_submission
from .reward import INVALID_REWARD, terminal_reward


class Evaluator:
    def __init__(self, truth: dict):
        self._truth = truth
        self.episode_id = truth["episode_id"]
        self.log = []  # журнал оценщика: ошибки структуры и диагностика; агенту не отдаётся

    def truth(self) -> dict:
        """Только для стороны проверяющего (oracle, тесты). Среда агенту это не передаёт."""
        return self._truth

    def grade_empty(self, inspection_steps: int) -> dict:
        """Усечённый эпизод (лимит шагов или времени): итоговый ответ пустой."""
        return self.grade({"episode_id": self.episode_id, "findings": []}, inspection_steps)

    def grade(self, payload, inspection_steps: int) -> dict:
        """Оценка итогового ответа. Любой payload, кроме корректного findings.json (в том числе None), неверен."""
        try:
            findings = validate_submission(payload, self.episode_id)
        except StructureError as error:
            self.log.append(f"invalid submission: {error}")
            return {"status": "invalid", "error": str(error), "tp": 0, "fp": 0, "fn": len(self._truth["issues"]),
                    "precision": 0.0, "recall": 0.0, "f1": 0.0, "reward": INVALID_REWARD,
                    "inspection_steps": inspection_steps, "matched_issue_ids": []}
        result = match_findings(findings, self._truth["issues"])
        return {"status": "graded", "error": None, "tp": result.tp, "fp": result.fp, "fn": result.fn,
                "precision": result.precision, "recall": result.recall, "f1": result.f1,
                "reward": terminal_reward(result.f1, inspection_steps), "inspection_steps": inspection_steps,
                "matched_issue_ids": result.matched_issue_ids}
