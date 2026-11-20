"""Мелкие модули: калькулятор, деньги, время."""
from datetime import datetime, timedelta, timezone

import pytest

from audit_env import money
from audit_env import timeutil as tu
from audit_env.calculator import CalcError, evaluate


@pytest.mark.parametrize("expr,expected", [
    ("1+1", "2"), ("2*(3+4)", "14"), ("10/4", "2.5"), ("-5+2", "-3"), ("0.1+0.2", "0.3"),
    ("1/3", "0.333333333333"), ("  7 - 2 * 3 ", "1"), ("--4", "4"), ("12.5*0.8", "10"),
    ("100/8", "12.5"), ("0", "0"), ("0-0", "0"),
])
def test_calculator_arithmetic(expr, expected):
    assert evaluate(expr) == expected


def test_calculator_is_exact_decimal_arithmetic():
    assert evaluate("0.1*3") == "0.3"
    assert evaluate("1000000000.01*3") == "3000000000.03"


@pytest.mark.parametrize("expr", [
    "__import__('os').system('dir')", "open('x')", "2**3", "1+", "(1+2", "1+2)", "abc", "1;2", "1,2", "1 2", "",
    "   ", "1/0", "1e5", "0x10", "5%2", "a=1", "[1]", "'a'*3", "1" * 40, "9" * 25 + "*" + "9" * 25, "()",
    "1+" * 200,
])
def test_calculator_rejects_everything_but_arithmetic(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_calculator_rejects_non_strings():
    for value in (None, 5, ["1"]):
        with pytest.raises(CalcError):
            evaluate(value)


def test_convert_rounding_modes():
    assert money.convert(1250, "EUR", "USD", "0.10000000", "half_up") == 125
    assert money.convert(105, "EUR", "USD", "0.50000000", "half_up") == 53   # 52.5 -> 53
    assert money.convert(105, "EUR", "USD", "0.50000000", "half_even") == 52  # 52.5 -> 52
    assert money.convert(-105, "EUR", "USD", "0.50000000", "half_up") == -53  # симметрично для возвратов


def test_convert_respects_currency_exponents():
    assert money.convert(100, "USD", "JPY", "150.00000000") == 150     # 1.00 USD -> 150 JPY (у иены нет копеек)
    assert money.convert(150, "JPY", "USD", "0.00666667") == 100      # 150 JPY -> 1.00 USD
    assert money.convert(5, "USD", "USD", "9.99") == 5                # та же валюта: курс не применяется


def test_rate_string_has_eight_digits():
    from decimal import Decimal
    assert money.rate_str(Decimal("0.5")) == "0.50000000"


def test_timestamps_need_an_offset():
    with pytest.raises(ValueError):
        tu.parse_ts("2026-03-01T10:00:00")
    assert tu.parse_ts("2026-03-01T10:00:00+05:00") == tu.parse_ts("2026-03-01T05:00:00+00:00")


def test_offsets_roundtrip():
    for text in ("+05:00", "-03:00", "+00:00", "+05:30"):
        assert tu.offset_text(tu.tz_from_offset(text)) == text


def test_fx_date_depends_on_the_rule():
    tz = tu.tz_from_offset("+05:00")
    ts = tu.parse_ts("2026-03-02T01:30:00+05:00")  # в UTC это ещё 1 марта
    assert tu.fx_date(ts, "company_date", tz) == "2026-03-02"
    assert tu.fx_date(ts, "utc_date", tz) == "2026-03-01"
    with pytest.raises(ValueError):
        tu.fx_date(ts, "nope", tz)


def test_period_side_compares_instants_not_strings():
    tz = timezone(timedelta(hours=5))
    start, end = datetime(2026, 3, 1, tzinfo=tz), datetime(2026, 4, 1, tzinfo=tz)
    assert tu.period_side(tu.parse_ts("2026-03-31T21:00:00+00:00"), start, end) == 1   # локально уже 1 апреля 02:00
    assert tu.period_side(tu.parse_ts("2026-03-31T18:59:59+00:00"), start, end) == 0
    assert tu.period_side(tu.parse_ts("2026-03-01T00:00:00+05:00"), start, end) == 0   # начало включено
    assert tu.period_side(tu.parse_ts("2026-04-01T00:00:00+05:00"), start, end) == 1   # конец не включён
    assert tu.period_side(tu.parse_ts("2026-02-28T18:59:59+00:00"), start, end) == -1
