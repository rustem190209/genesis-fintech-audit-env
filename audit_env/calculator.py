"""Безопасный калькулятор: только + - * / скобки и десятичные числа. Без eval, кода и файлов."""
import re
from decimal import Decimal, DecimalException, localcontext

MAX_LEN = 200
MAX_DIGITS = 30
MAX_RESULT = Decimal(10) ** 15  # результат должен помещаться в 12 знаков после точки без потери
_TOKEN = re.compile(r"\s*(?:([0-9]+(?:\.[0-9]+)?)|(.))")  # только цифры ASCII, без юникодных


class CalcError(ValueError):
    pass


def _tokenize(expr: str):
    tokens, pos = [], 0
    while pos < len(expr):
        match = _TOKEN.match(expr, pos)
        if not match:
            break
        number, symbol = match.group(1), match.group(2)
        if number is not None:
            if len(number.replace(".", "")) > MAX_DIGITS:
                raise CalcError("number too long")
            tokens.append(("num", Decimal(number)))
        elif symbol in "+-*/()":
            tokens.append(("op", symbol))
        else:
            raise CalcError("unsupported character")
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def take(self):
        token = self.peek()
        self.pos += 1
        return token

    def expr(self):  # сложение и вычитание
        value = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            right = self.term()
            value = value + right if op == "+" else value - right
        return value

    def term(self):  # умножение и деление
        value = self.factor()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            right = self.factor()
            value = value * right if op == "*" else value / right
        return value

    def factor(self):
        kind, val = self.take()
        if kind == "num":
            return val
        if (kind, val) == ("op", "-"):
            return -self.factor()
        if (kind, val) == ("op", "+"):
            return self.factor()
        if (kind, val) == ("op", "("):
            value = self.expr()
            if self.take() != ("op", ")"):
                raise CalcError("missing )")
            return value
        raise CalcError("syntax error")


def evaluate(expression: str) -> str:
    """Возвращает результат строкой без экспоненты (до 12 знаков после точки).
    Любая арифметическая проблема превращается в CalcError, а не в падение."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalcError("empty expression")
    if len(expression) > MAX_LEN:
        raise CalcError("expression too long")
    parser = _Parser(_tokenize(expression.strip()))
    with localcontext() as context:  # свои настройки: результат не зависит от глобального контекста decimal
        context.prec = 60
        try:
            value = parser.expr()
            if parser.pos != len(parser.tokens):
                raise CalcError("syntax error")
            if abs(value) > MAX_RESULT:
                raise CalcError("result too large")
            text = format(value.quantize(Decimal("0.000000000001")).normalize(), "f")
        except DecimalException:  # деление на ноль, переполнение и т. п.
            raise CalcError("arithmetic error")
    return text if text != "-0" else "0"
