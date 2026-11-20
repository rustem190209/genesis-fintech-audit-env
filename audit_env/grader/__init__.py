"""Оценщик: проверка структуры выводов, сопоставление с эталоном, награда, проверка данных."""
from .matcher import MatchResult, StructureError, match_findings, validate_submission
from .reward import INVALID_REWARD, STEP_COST, f1_score, terminal_reward

__all__ = ["MatchResult", "StructureError", "match_findings", "validate_submission",
           "INVALID_REWARD", "STEP_COST", "f1_score", "terminal_reward"]
