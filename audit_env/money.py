"""Деньги: целые минимальные единицы, разрядность валют, конвертация по курсу."""
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, localcontext

# Сколько знаков после запятой у валюты (JPY — 0)
EXPONENT = {"USD": 2, "EUR": 2, "GBP": 2, "KZT": 2, "JPY": 0}

ROUNDING_MODES = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN}


def convert(amount_minor: int, src: str, dst: str, rate: str, rounding: str = "half_up") -> int:
    """Перевод суммы из src в dst по курсу rate (1 src = rate dst). Результат в минимальных единицах dst."""
    if src == dst:
        return int(amount_minor)
    with localcontext() as context:  # результат не зависит от глобальных настроек decimal
        context.prec = 50
        major = Decimal(int(amount_minor)) / (Decimal(10) ** EXPONENT[src])
        value = major * Decimal(rate) * (Decimal(10) ** EXPONENT[dst])
        return int(value.quantize(Decimal(1), rounding=ROUNDING_MODES[rounding]))


def rate_str(value: Decimal) -> str:
    """Курс всегда строкой с 8 знаками (чтобы не терять точность во float)."""
    return format(value.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_EVEN), "f")
