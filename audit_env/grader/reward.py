"""Метрики и терминальная награда (общая функция из раздела 05 кейса)."""
STEP_COST = 0.002
INVALID_REWARD = -1.0


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if tp + fp > 0 else 0.0  # при нулевом знаменателе P = 0


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if tp + fn > 0 else 0.0  # при нулевом знаменателе R = 0


def f1_score(tp: int, fp: int, fn: int) -> float:
    """F1 = 2TP/(2TP+FP+FN). Если эталон и выводы пусты, F1 = 1. Если эталон пуст, а выводы есть, F1 = 0."""
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else 2 * tp / denominator


def terminal_reward(f1: float, inspection_steps: int) -> float:
    """clip(2*F1 - 1 - 0.002*inspection_steps, -1, 1). inspection_steps = шаги до submit."""
    return max(-1.0, min(1.0, 2 * f1 - 1 - STEP_COST * inspection_steps))
