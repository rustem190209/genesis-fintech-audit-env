"""Генератор эпизодов: скрытая история компании -> документы и проводки -> внесённые ошибки.

Порядок такой:
1. Строим согласованную «правильную» историю (счета, платежи, курсы, журнал).
2. Добавляем отвлекающие, но корректные случаи.
3. Вносим ошибки с записанным причинным изменением (provenance).
4. Проводки пишем дважды: «чистые» (без ошибок) и фактические. Разница = ошибки.
5. Эталон (truth) строится из внесённых изменений, а не из «подозрительных строк».

Вся случайность идёт через random.Random: история (rng «h») не зависит от варианта
представления (rng «p»: ID, порядок, смещения времени). Поэтому тот же config с другим
variant даёт ту же историю с другими ID и порядком (парные проверки).
"""
import copy
import hashlib
import json
import os
import random
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Context, Decimal, localcontext

from . import money
from . import timeutil as tu

GEN_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
ERROR_TYPES = ("duplicate_invoice", "payment_allocation", "fx_error", "period_cutoff")
INTERACTIONS = ("fx_boundary_partial", "dup_cutoff")
# Режимы, которые можно задать явно (config.error_modes); "random" — выбор генератора
ERROR_MODES = {"payment_allocation": ("random", "missing", "partial"),
               "fx_error": ("random", "wrong_date", "wrong_rate"),
               "period_cutoff": ("random", "end_late", "end_early", "start_late", "start_early")}
# Допуск суммы τ по типам (определения проверок: по умолчанию 0). Все суммы этой среды считаются точно:
# итог счёта, остаток в UNAPPLIED и разница пересчёта по правилу округления из policy.json.
TOLERANCE = {"duplicate_invoice": 0, "payment_allocation": 0, "fx_error": 0, "period_cutoff": 0}
MAX_ISSUES = 12
MIN_LEDGER, MAX_LEDGER, MAX_DOCUMENTS = 200, 2000, 200  # предложенные кейсом размеры эпизода
ADJ_DAYS = 5  # сколько дней вокруг периода попадает в журнал (контекст)
UTC = timezone.utc

CURRENCY_POOL = ["USD", "EUR", "KZT", "GBP", "JPY"]
APPROX_VALUE = {"USD": Decimal("1"), "EUR": Decimal("1.08"), "GBP": Decimal("1.27"),
                "KZT": Decimal("0.0021"), "JPY": Decimal("0.0067")}
AMOUNT_RANGE = {"USD": (200, 20000), "EUR": (200, 18000), "GBP": (200, 15000),
                "KZT": (100000, 9000000), "JPY": (30000, 3000000)}
COMPANY_OFFSETS = [5, 6, 3, -5, 1, -3, 8]  # без 0: дата в UTC и локальная должны расходиться
DOC_OFFSETS = [0, 1, 3, 5, 6, 8, -3, -5]
SERVICES = ["Consulting", "Hosting", "Licence", "Maintenance", "Delivery", "Design work", "Training", "Support"]

LEDGER_COLUMNS = ["row_id", "timestamp", "account", "debit_minor", "credit_minor",
                  "currency", "entity_id", "invoice_id", "payment_id"]


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- конфигурация

def default_config() -> dict:
    return {
        "n_invoices": 60,
        "n_journal": 100,
        "base_currency": None,
        "errors": {t: 0 for t in ERROR_TYPES},
        "interactions": [],
        "distractors": {"split_payments": 2, "refunds": 2, "recurring": 2, "deposits": 2,
                        "tz_boundary": 3, "fx_clean_near_midnight": 2, "repeated_amounts": 2,
                        "multi_invoice_payments": 2, "fx_partial": 2},
        "error_modes": {name: "random" for name in ERROR_MODES},
        "variant": {"id_salt": "", "shuffle": False},
    }


def _check_int(value, name, low, high):
    """Целое в границах. bool не считается числом: True и 1 дали бы разные эпизоды при одинаковом смысле."""
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def merge_config(config: dict | None) -> dict:
    """Дополняет config значениями по умолчанию и проверяет его. Неизвестные ключи и неверные типы запрещены."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("config must be an object or null")
    cfg = default_config()
    for key, value in (config or {}).items():
        if key not in cfg:
            raise ValueError(f"unknown config key: {key}")
        if isinstance(cfg[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"config.{key} must be an object")
            for sub, subval in value.items():
                if sub not in cfg[key]:
                    raise ValueError(f"unknown config key: {key}.{sub}")
                cfg[key][sub] = subval
        else:
            cfg[key] = value
    for t in ERROR_TYPES:
        _check_int(cfg["errors"][t], f"errors.{t}", 0, MAX_ISSUES)
    for name, count in cfg["distractors"].items():
        _check_int(count, f"distractors.{name}", 0, 10)
    interactions = cfg["interactions"]
    if (not isinstance(interactions, list) or any(x not in INTERACTIONS for x in interactions)
            or len(set(interactions)) != len(interactions)):
        raise ValueError(f"interactions must be a list of unique values from {INTERACTIONS}")
    if sum(cfg["errors"].values()) + 2 * len(interactions) > MAX_ISSUES:
        raise ValueError(f"too many issues (max {MAX_ISSUES})")
    for name, mode in cfg["error_modes"].items():
        if mode not in ERROR_MODES[name]:
            raise ValueError(f"error_modes.{name} must be one of {ERROR_MODES[name]}")
    _check_int(cfg["n_invoices"], "n_invoices", 20, 80)
    _check_int(cfg["n_journal"], "n_journal", 0, 800)
    if cfg["base_currency"] not in (None, "USD", "EUR", "KZT"):
        raise ValueError("base_currency must be null, USD, EUR or KZT")
    variant = cfg["variant"]
    if (not isinstance(variant.get("shuffle"), bool) or not isinstance(variant.get("id_salt"), str)
            or len(variant["id_salt"]) > 64):
        raise ValueError("variant must be {id_salt: str (up to 64 chars), shuffle: bool}")
    return cfg


# ---------------------------------------------------------------- результат

class Episode:
    """Готовый эпизод. Открытые данные: manifest, policy, ledger, invoices, payments, fx.
    Закрытые данные: truth, provenance (агент их не видит)."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def open_files(self) -> dict:
        """Содержимое открытых файлов как текст (ключ = имя файла)."""
        def jsonl(rows):
            return "".join(canon(r) + "\n" for r in rows)

        def cell(value):
            return "null" if value is None else str(value)

        ledger = [",".join(LEDGER_COLUMNS)] + [",".join(cell(r[c]) for c in LEDGER_COLUMNS) for r in self.ledger]
        fx = ["date,source_currency,target_currency,rate"] + [
            f"{r['date']},{r['source_currency']},{r['target_currency']},{r['rate']}" for r in self.fx]
        return {
            "manifest.json": json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            "policy.json": json.dumps(self.policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            "ledger.csv": "\n".join(ledger) + "\n",
            "invoices.jsonl": jsonl(self.invoices),
            "payments.jsonl": jsonl(self.payments),
            "fx.csv": "\n".join(fx) + "\n",
        }

    def closed_files(self) -> dict:
        return {"truth.json": json.dumps(self.truth, indent=2, sort_keys=True) + "\n",
                "provenance.json": json.dumps(self.provenance, indent=2, sort_keys=True) + "\n"}


# ---------------------------------------------------------------- построитель

class _Builder:
    def __init__(self, seed: int, cfg: dict):
        self.cfg = cfg
        hist_key = canon({"v": GEN_VERSION, "seed": seed, "cfg": {k: v for k, v in cfg.items() if k != "variant"}})
        # Секрет проверяющего (переменная окружения) входит в ключ всех случайных чисел. Без него
        # агент не восстановит seed ни по episode_id, ни перебором по открытым данным: генератор открыт,
        # но без секрета он строит другие компании. Одинаковые seed + config + секрет дают те же данные.
        secret = os.environ.get("AUDIT_ENV_SECRET", "")
        if secret:
            hist_key += "|secret:" + _digest(secret)
        pres_key = hist_key + "|pres|" + canon(cfg["variant"])
        self.episode_id = "ep-" + _digest(hist_key + canon(cfg["variant"]))[:12]
        self.h = random.Random(int(_digest(hist_key)[:16], 16))
        self.p = random.Random(int(_digest(pres_key)[:16], 16))
        self.seed = seed

        h = self.h
        self.base = cfg["base_currency"] or h.choice(["USD", "EUR", "KZT"])
        self.foreign = h.sample([c for c in CURRENCY_POOL if c != self.base], 2)
        self.company_tz = timezone(timedelta(hours=h.choice(COMPANY_OFFSETS)))
        month = h.randint(1, 12)
        year_end, month_end = (2027, 1) if month == 12 else (2026, month + 1)
        self.start = datetime(2026, month, 1, tzinfo=self.company_tz).astimezone(UTC)
        self.end = datetime(year_end, month_end, 1, tzinfo=self.company_tz).astimezone(UTC)
        self.fx_rule = h.choice(["company_date", "utc_date"])
        self.other_rule = "utc_date" if self.fx_rule == "company_date" else "company_date"
        self.rounding = h.choice(["half_up", "half_even"])

        self.serial = 0       # счётчик для уникальных номеров заказов в отвлекающих счетах
        self.fx = {}          # (валюта, дата) -> курс-строка к базовой валюте
        self.entities = {}    # key -> {"currency"}
        self.invoices, self.payments, self.journal, self.plants = [], [], [], []
        self.inv_by_key, self.pay_by_key = {}, {}
        self._build_fx()
        self._build_entities()

    # ---- мелкие помощники
    def rand_ts(self, lo, hi):
        secs = int((hi - lo).total_seconds())
        return lo + timedelta(seconds=self.h.randint(0, max(secs, 1)))

    def clear_of_boundary(self, ts):
        """Обычные записи не ставим ближе 90 минут к границе периода: сторона должна быть однозначной."""
        for boundary in (self.start, self.end):
            if abs(ts - boundary) < timedelta(minutes=90):
                ts = boundary + timedelta(seconds=self.h.randint(2 * 3600, 3 * 3600))
        return ts

    def secs(self, low_hours, high_hours):
        """Случайный сдвиг с точностью до секунды: внесённые ошибки не должны выделяться «ровным часом»."""
        return timedelta(seconds=self.h.randint(int(low_hours * 3600), int(high_hours * 3600)))

    def inside_ts(self, from_start_days, to_end_days):
        lo = self.start + timedelta(days=from_start_days)
        hi = self.end - timedelta(days=to_end_days)
        return self.clear_of_boundary(self.rand_ts(lo, hi))

    def local_date(self, ts):
        return ts.astimezone(self.company_tz).date()

    def rate_for(self, cur, ts, rule=None):
        date = tu.fx_date(ts, rule or self.fx_rule, self.company_tz)
        return self.fx[(cur, date)], f"{date}:{cur}:{self.base}"

    def rand_total(self, cur):
        low, high = AMOUNT_RANGE[cur]
        exp = money.EXPONENT[cur]
        return self.h.randint(low, high) * 10 ** exp + (self.h.randint(0, 99) if exp == 2 else 0)

    def make_lines(self, total, issued):
        """Строки счёта. Шаблоны описаний общие для всех счетов (обычных, ловушек и ошибок),
        чтобы вид текста не выдавал, к какой группе относится счёт."""
        local = issued.astimezone(self.company_tz)
        label, week = local.strftime("%Y-%m"), local.isocalendar()[1]
        count = self.h.randint(1, 3)
        cuts = sorted(self.h.sample(range(1, total), count - 1))  # разные точки: нулевых строк не будет
        parts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
        lines = []
        for svc, amount in zip(self.h.sample(SERVICES, count), parts):
            style = self.h.random()
            if style < 0.6:
                text = f"{svc} {label}"
            elif style < 0.8:
                text = f"{svc}, week {week}"
            else:
                self.serial += 1
                text = f"{svc} order {self.serial}"
            lines.append({"description": text, "quantity": self.h.randint(1, 5), "amount_minor": amount})
        return lines

    def other_lines(self, total, issued, avoid):
        """Строки, гарантированно отличные от avoid (законные одинаковые суммы не должны стать дублем)."""
        lines = self.make_lines(total, issued)
        while lines == avoid:
            lines = self.make_lines(total, issued)
        return lines

    def daytime_ts(self, local_day):
        """Мгновение днём по местному времени (8:00-20:00), с точностью до секунды."""
        start = datetime(local_day.year, local_day.month, local_day.day, tzinfo=self.company_tz)
        return self.clear_of_boundary((start + self.secs(8, 20)).astimezone(UTC))

    def mismatch_ts(self, local_day):
        """Мгновение, у которого локальная дата и дата в UTC разные (с точностью до секунды)."""
        hours = int(self.company_tz.utcoffset(None).total_seconds() // 3600)
        if hours > 0:
            seconds = self.h.randint(600, hours * 3600 - 600)
        else:
            seconds = self.h.randint((24 + hours) * 3600 + 600, 24 * 3600 - 600)
        ts = datetime(local_day.year, local_day.month, local_day.day, tzinfo=self.company_tz) + timedelta(seconds=seconds)
        ts = ts.astimezone(UTC)
        assert tu.fx_date(ts, "company_date", self.company_tz) != tu.fx_date(ts, "utc_date", self.company_tz)
        return ts

    # ---- курсы и участники
    def _build_fx(self):
        first = self.local_date(self.start) - timedelta(days=ADJ_DAYS + 7)
        last = self.local_date(self.end) + timedelta(days=ADJ_DAYS + 7)
        for cur in self.foreign:
            value = APPROX_VALUE[cur] / APPROX_VALUE[self.base] * Decimal(str(round(1 + self.h.uniform(-0.03, 0.03), 6)))
            for i in range((last - first).days + 1):
                self.fx[(cur, (first + timedelta(days=i)).isoformat())] = money.rate_str(value)
                value *= Decimal(str(round(1 + self.h.gauss(0, 0.004), 6)))

    def _build_entities(self):
        for i in range(max(6, self.cfg["n_invoices"] // 5)):
            cur = self.base if self.h.random() < 0.6 else self.h.choice(self.foreign)
            if i == 0:
                cur = self.base  # гарантия: есть клиент в базовой валюте
            elif i == 1:
                cur = self.foreign[0]  # и клиент в иностранной
            self.entities[f"ent:{i}"] = {"currency": cur}

    def entity(self, currency=None):
        keys = [k for k, v in self.entities.items() if currency is None or v["currency"] == currency]
        return self.h.choice(keys)

    # ---- создание объектов
    def add_invoice(self, entity, issued, total=None, lines=None, **extra):
        cur = self.entities[entity]["currency"]
        total = total if total is not None else self.rand_total(cur)
        n = len(self.invoices)
        inv = {"key": f"inv:{n}", "doc": f"doc:inv:{n}", "entity": entity, "issued": issued, "currency": cur,
               "total": total, "lines": lines if lines is not None else self.make_lines(total, issued),
               "post_ts": None, "tag": "ordinary"}
        inv.update(extra)
        self.invoices.append(inv)
        self.inv_by_key[inv["key"]] = inv
        return inv

    def add_payment(self, ts, cur, amount, refs, entity, tag="ordinary", plan=None):
        n = len(self.payments)
        pay = {"key": f"pay:{n}", "doc": f"doc:pay:{n}", "ts": ts, "currency": cur, "amount": amount,
               "refs": refs, "entity": entity, "plan": plan, "tag": tag}
        self.payments.append(pay)
        self.pay_by_key[pay["key"]] = pay
        return pay

    def pay_amount_for(self, total_base, cur, ts):
        """Сколько нужно заплатить в валюте cur, чтобы покрыть счёт в базовой валюте."""
        rate, _ = self.rate_for(cur, ts)
        major = Decimal(total_base) / Decimal(10) ** money.EXPONENT[self.base] / Decimal(rate)
        return int((major * Decimal(10) ** money.EXPONENT[cur]).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    # ---- обычная история
    def build_ordinary(self):
        for _ in range(self.cfg["n_invoices"]):
            issued = self.clear_of_boundary(self.rand_ts(self.start - timedelta(days=3), self.end + timedelta(days=3)))
            inv = self.add_invoice(self.entity(), issued)
            self.settle_ordinary(inv)

    def settle_ordinary(self, inv):
        if self.h.random() < 0.22:
            return  # счёт ещё не оплачен
        ts = inv["issued"] + timedelta(days=self.h.uniform(1, 35))
        if ts >= self.end + timedelta(days=ADJ_DAYS - 1):
            return
        ts = self.clear_of_boundary(ts.replace(microsecond=0))
        if inv["currency"] == self.base and self.h.random() < 0.18:
            cur = self.h.choice(self.foreign)
            amount = self.pay_amount_for(inv["total"], cur, ts)
            self.add_payment(ts, cur, amount, [(inv["key"], amount)], inv["entity"])
            return
        cur, amount = inv["currency"], inv["total"]
        self.add_payment(ts, cur, amount, [(inv["key"], amount)], inv["entity"])

    def build_journal(self):
        for _ in range(self.cfg["n_journal"]):
            cur = self.base if self.h.random() < 0.8 else self.h.choice(self.foreign)
            low, high = AMOUNT_RANGE[cur]
            amount = max(1, self.h.randint(low // 20, high // 10)) * 10 ** money.EXPONENT[cur]
            ts = self.rand_ts(self.start - timedelta(days=ADJ_DAYS), self.end + timedelta(days=ADJ_DAYS))
            self.journal.append({"ts": ts.replace(microsecond=0), "currency": cur, "amount": amount})

    # ---- отвлекающие, но корректные случаи
    def build_distractors(self):
        counts = self.cfg["distractors"]
        for _ in range(counts["split_payments"]):
            inv = self.add_invoice(self.entity(), self.inside_ts(1, 14), tag="split")
            parts = self.h.randint(2, 3)
            cuts = sorted(self.h.sample(range(1, inv["total"]), parts - 1))
            amounts = [b - a for a, b in zip([0] + cuts, cuts + [inv["total"]])]
            ts = inv["issued"]
            for amount in amounts:
                ts = self.clear_of_boundary((ts + timedelta(days=self.h.uniform(1, 6))).replace(microsecond=0))
                self.add_payment(ts, inv["currency"], amount, [(inv["key"], amount)], inv["entity"], tag="split")
        for _ in range(counts["refunds"]):
            inv = self.add_invoice(self.entity(), self.inside_ts(1, 14), tag="refund")
            ts = self.clear_of_boundary((inv["issued"] + timedelta(days=self.h.uniform(1, 4))).replace(microsecond=0))
            self.add_payment(ts, inv["currency"], inv["total"], [(inv["key"], inv["total"])], inv["entity"], tag="refund")
            back = -self.h.randint(1, inv["total"] // 2)
            ts2 = self.clear_of_boundary((ts + timedelta(days=self.h.uniform(1, 5))).replace(microsecond=0))
            self.add_payment(ts2, inv["currency"], back, [(inv["key"], back)], inv["entity"], tag="refund")
        for _ in range(counts["recurring"]):
            ent = self.entity()
            total = self.rand_total(self.entities[ent]["currency"])
            first = self.inside_ts(0, 22)
            for week in range(3):
                service_at = first + timedelta(days=7 * week)  # неделя услуги: у каждого счёта своя
                issued = self.clear_of_boundary(service_at + self.secs(-3, 3))  # время выставления чуть плавает
                iso_week = service_at.astimezone(self.company_tz).isocalendar()[1]
                lines = [{"description": f"Support, week {iso_week}", "quantity": 1, "amount_minor": total}]
                inv = self.add_invoice(ent, issued, total=total, lines=lines, tag="recurring")
                self.settle_ordinary(inv)
        for _ in range(counts["deposits"]):
            ent = self.entity()
            cur = self.entities[ent]["currency"]
            ts = self.inside_ts(0, 1)
            self.add_payment(ts.replace(microsecond=0), cur, self.rand_total(cur), [], ent, tag="deposit")
        for i in range(counts["tz_boundary"]):
            boundary = self.end if i % 2 == 0 else self.start
            sign = -1 if (i // 2) % 2 == 0 else 1
            issued = (boundary + sign * self.secs(0.17, 3)).replace(microsecond=0)
            self.add_invoice(self.entity(), issued, tag="tz_boundary")
        for _ in range(counts["fx_clean_near_midnight"]):
            inv = self.add_invoice(self.entity(self.base), self.inside_ts(1, 12), tag="fx_clean")
            day = self.local_date(inv["issued"]) + timedelta(days=self.h.randint(1, 6))
            ts = self.mismatch_ts(day)
            cur = self.h.choice(self.foreign)
            amount = self.pay_amount_for(inv["total"], cur, ts)
            self.add_payment(ts, cur, amount, [(inv["key"], amount)], inv["entity"], tag="fx_clean")
        for _ in range(counts["fx_partial"]):
            # законно: часть счёта оплачена в чужой валюте, кредит AR заметно меньше суммы счёта
            inv = self.add_invoice(self.entity(self.base), self.inside_ts(1, 14), tag="fx_partial")
            cur = self.h.choice(self.foreign)
            day = self.local_date(inv["issued"]) + timedelta(days=self.h.randint(1, 5))
            ts = self.mismatch_ts(day) if self.h.random() < 0.5 else self.daytime_ts(day)
            share = self.h.uniform(0.3, 0.8)
            amount = self.pay_amount_for(int(inv["total"] * share), cur, ts)
            self.add_payment(ts, cur, amount, [(inv["key"], amount)], inv["entity"], tag="fx_partial")
            if self.h.random() < 0.6:  # остаток доплачен в базовой валюте
                rest = inv["total"] - int(inv["total"] * share)
                ts2 = self.clear_of_boundary((ts + timedelta(days=self.h.uniform(1, 5))).replace(microsecond=0))
                self.add_payment(ts2, self.base, rest, [(inv["key"], rest)], inv["entity"], tag="fx_partial")
        for _ in range(counts["repeated_amounts"]):
            total = self.rand_total(self.base)
            previous = None
            for _ in range(2):  # одна сумма у двух счетов; состав строк разный, это не дубль
                ent = self.entity(self.base)
                issued = self.inside_ts(0, 2)
                lines = self.other_lines(total, issued, previous)
                previous = lines
                self.settle_ordinary(self.add_invoice(ent, issued, total=total, lines=lines, tag="same_amount"))
            ent = self.entity()
            total2 = self.rand_total(self.entities[ent]["currency"])
            first = self.inside_ts(0, 16)
            # один клиент, одна сумма, близкое время (как у дубля), но другие услуги
            first_lines = self.make_lines(total2, first)
            self.add_invoice(ent, first, total=total2, lines=first_lines, tag="same_amount")
            second = self.clear_of_boundary(first + self.secs(2, 60))
            self.add_invoice(ent, second, total=total2, lines=self.other_lines(total2, second, first_lines), tag="same_amount")
        for _ in range(counts["multi_invoice_payments"]):
            ent = self.entity()
            cur = self.entities[ent]["currency"]
            first = self.add_invoice(ent, self.inside_ts(0, 14), tag="multi")
            second_at = self.clear_of_boundary(first["issued"] + self.secs(20, 72))
            second = self.add_invoice(ent, second_at, tag="multi")
            ts = self.clear_of_boundary((second["issued"] + timedelta(days=self.h.uniform(1, 5))).replace(microsecond=0))
            self.add_payment(ts, cur, first["total"] + second["total"],
                             [(first["key"], first["total"]), (second["key"], second["total"])], ent, tag="multi")

    # ---- внесение ошибок
    def plant_errors(self):
        for _ in range(self.cfg["errors"]["duplicate_invoice"]):
            self.plant_duplicate()
        for _ in range(self.cfg["errors"]["payment_allocation"]):
            self.plant_payment_error(alloc=True, fx=False)
        for _ in range(self.cfg["errors"]["fx_error"]):
            self.plant_payment_error(alloc=False, fx=True)
        for _ in range(self.cfg["errors"]["period_cutoff"]):
            self.plant_cutoff()
        for name in self.cfg["interactions"]:
            if name == "fx_boundary_partial":
                self.plant_payment_error(alloc=True, fx=True, boundary=True, inter=name)
            else:
                self.plant_duplicate(cutoff=True, inter=name)

    def plant_duplicate(self, cutoff=False, inter=None):
        ent = self.entity()
        if cutoff:
            orig_at = self.end - self.secs(30, 60)
            dup_at = self.end - self.secs(3, 20)
        else:
            orig_at = self.inside_ts(1, 8)
            dup_at = self.clear_of_boundary(orig_at + self.secs(2, 60))
        orig = self.add_invoice(ent, orig_at.replace(microsecond=0), tag="dup_original")
        extra = {}
        if cutoff:  # дубль проведён уже после конца периода
            extra["post_ts"] = self.end + self.secs(3, 60)
        dup = self.add_invoice(ent, dup_at.replace(microsecond=0), total=orig["total"],
                               lines=copy.deepcopy(orig["lines"]), tag="dup", dup_of=orig["key"], **extra)
        self.plants.append({"kind": "duplicate_invoice", "orig": orig["key"], "dup": dup["key"], "inter": inter})
        if cutoff:
            self.plants.append({"kind": "period_cutoff", "inv": dup["key"], "variant": "end_late", "inter": inter})

    def plant_payment_error(self, alloc, fx, boundary=False, inter=None):
        """Платёж с ошибкой распределения и/или курса. Два флага сразу = взаимодействующие ошибки."""
        if fx:
            hours = int(self.company_tz.utcoffset(None).total_seconds() // 3600)
            if boundary and hours > 0:  # платёж в первый день периода: счёт выставлен чуть раньше, в прошлом периоде
                issued = (self.start - self.secs(30, 70)).replace(microsecond=0)
            else:
                issued = self.inside_ts(2, 12)
            inv = self.add_invoice(self.entity(self.base), issued, tag="fx_error")
            cur = self.h.choice(self.foreign)
            if boundary:  # сценарий «на границе»: дата по UTC и по поясу компании разные
                day = self.local_date(self.start if hours > 0 else self.end - timedelta(seconds=1))
                ts = self.mismatch_ts(day)
            else:
                day = self.local_date(inv["issued"]) + timedelta(days=self.h.randint(1, 6))
                ts = self.mismatch_ts(day) if self.h.random() < 0.5 else self.daytime_ts(day)
            share = 1.0 if self.h.random() < 0.5 else self.h.uniform(0.3, 0.8)  # бывает и частичная оплата
            amount = self.pay_amount_for(int(inv["total"] * share), cur, ts)
        else:
            inv = self.add_invoice(self.entity(), self.inside_ts(1, 14), tag="alloc_error")
            ts = self.clear_of_boundary((inv["issued"] + timedelta(days=self.h.uniform(1, 6))).replace(microsecond=0))
            cur, amount = inv["currency"], inv["total"]
        plan = {"alloc": alloc, "fx": fx}
        if alloc:
            wanted = self.cfg["error_modes"]["payment_allocation"]
            plan["mode"] = self.h.choice(["missing", "partial"]) if wanted == "random" else wanted
            plan["fraction"] = self.h.uniform(0.3, 0.75)
        if fx:
            good = Decimal(self.rate_for(cur, ts)[0])
            wanted = self.cfg["error_modes"]["fx_error"]
            wrong_date = self.pick_wrong_date(cur, ts, good) if wanted != "wrong_rate" else None
            use_date = wrong_date and (wanted == "wrong_date" or self.h.random() < 0.7)
            if use_date:  # курс взят на «не ту» дату (реальный курс из fx.csv)
                plan["wrong_rate"], plan["fx_variant"], plan["wrong_date"] = self.fx[(cur, wrong_date)], "wrong_date", wrong_date
            else:  # курс с опечаткой
                factor = 1 + self.h.choice([-1, 1]) * self.h.uniform(0.01, 0.04)
                plan["wrong_rate"], plan["fx_variant"] = money.rate_str(good * Decimal(str(round(factor, 6)))), "wrong_rate"
        pay = self.add_payment(ts, cur, amount, [(inv["key"], amount)], inv["entity"], tag="payment_error", plan=plan)
        self.plants.append({"kind": "payment_plan", "pay": pay["key"], "alloc": alloc, "fx": fx, "inter": inter})

    def pick_wrong_date(self, cur, ts, good):
        """Дата, курс которой заметно (>= 0.3%) отличается от верного: дата по другому правилу или
        одна из соседних (до 10 дней раньше или позже). Таблицу курсов не трогаем: в fx.csv не должно
        быть искусственных скачков."""
        correct = tu.fx_date(ts, self.fx_rule, self.company_tz)
        candidates = []
        other = tu.fx_date(ts, self.other_rule, self.company_tz)
        if other != correct:
            candidates.append(other)
        nearby = [(date.fromisoformat(correct) + timedelta(days=k)).isoformat() for k in range(-10, 11) if k]
        self.h.shuffle(nearby)
        for day in candidates + nearby:
            if abs(Decimal(self.fx[(cur, day)]) / good - 1) >= Decimal("0.003"):
                return day
        return None

    def plant_cutoff(self):
        wanted = self.cfg["error_modes"]["period_cutoff"]
        variant = self.h.choice(["end_late", "end_early", "start_late", "start_early"]) if wanted == "random" else wanted
        shift = self.secs
        if variant == "end_late":      # счёт в периоде, проведён после конца
            issued, posted = self.end - shift(3, 40), self.end + shift(3, 60)
        elif variant == "end_early":   # счёт за периодом, проведён раньше конца
            issued, posted = self.end + shift(3, 40), self.end - shift(3, 40)
        elif variant == "start_late":  # счёт в периоде, проведён до начала
            issued, posted = self.start + shift(3, 40), self.start - shift(3, 60)
        else:                          # счёт до периода, проведён внутри периода
            issued, posted = self.start - shift(3, 40), self.start + shift(3, 60)
        inv = self.add_invoice(self.entity(), issued.replace(microsecond=0), tag="cutoff", post_ts=posted.replace(microsecond=0))
        self.plants.append({"kind": "period_cutoff", "inv": inv["key"], "variant": variant, "inter": None})

    # ---- проводки
    def _payment_rows(self, pay, apply_plan):
        """Проводки платежа: список (роль, счёт, дебет, кредит, валюта, счёт-ключ) и сведения для эталона."""
        amount, cur = pay["amount"], pay["currency"]
        rows = [("cash", "CASH", max(amount, 0), max(-amount, 0), cur, None)]
        if not pay["refs"]:  # депозит без привязки к счёту: законно лежит в UNAPPLIED
            rows.append(("un", "UNAPPLIED", 0, amount, cur, None))
            return rows, {}
        plan = pay["plan"] if apply_plan else None
        if len(pay["refs"]) > 1:
            for inv_key, part in pay["refs"]:
                rows.append(("ar", "AR", 0, part, cur, inv_key))
            return rows, {}
        inv_key, part = pay["refs"][0]
        leg = self.inv_by_key[inv_key]["currency"]
        correct = part
        fx_key = None
        if leg != cur:
            rate, fx_key = self.rate_for(cur, pay["ts"])
            correct = money.convert(part, cur, leg, rate, self.rounding)
        booked = correct
        if plan and plan["fx"]:
            booked = money.convert(part, cur, leg, plan["wrong_rate"], self.rounding)
        applied, unapplied = booked, 0
        if plan and plan["alloc"]:
            applied = 0 if plan["mode"] == "missing" else int(round(booked * plan["fraction"]))
            unapplied = booked - applied
        if applied:
            rows.append(("ar", "AR", max(-applied, 0), max(applied, 0), leg, inv_key))
        if unapplied:
            rows.append(("un", "UNAPPLIED", 0, unapplied, leg, None))
        return rows, {"leg": leg, "correct": correct, "booked": booked, "unapplied": unapplied, "fx_key": fx_key}

    def posting_delay(self, issued):
        """Законная задержка проводки счёта: обычно минуты, но бывает на часы и дни позже или раньше.
        Главное правило: проводка остаётся в том же периоде, что и счёт. Поэтому «проводка не в момент
        счёта» сама по себе ошибкой не является, нужно сравнивать периоды."""
        if any(abs(issued - b) < timedelta(hours=1) for b in (self.start, self.end)):
            return timedelta(0)
        roll = self.h.random()
        if roll < 0.6:
            delay = timedelta(seconds=self.h.randint(0, 900))
        elif roll < 0.88:
            delay = self.secs(1, 72)       # проведён позже
        else:
            delay = -self.secs(1, 48)      # проведён заранее
        side = tu.period_side(issued, self.start, self.end)
        if tu.period_side(issued + delay, self.start, self.end) != side:
            delay = timedelta(seconds=self.h.randint(0, 900))
            if tu.period_side(issued + delay, self.start, self.end) != side:
                delay = timedelta(0)
        return delay

    def post_all(self):
        """Пишем проводки: фактические (с ошибками) и чистые (для проверки данных)."""
        self.rows, self.clean_rows = [], []

        def add(target, ts, account, debit, credit, cur, entity, inv, pay):
            row = {"key": f"row:{len(self.rows) + len(self.clean_rows)}", "ts": ts, "account": account,
                   "debit": debit, "credit": credit, "currency": cur, "entity": entity, "invoice": inv, "payment": pay}
            target.append(row)
            return row["key"]

        for inv in self.invoices:
            clean_ts = inv["issued"] + self.posting_delay(inv["issued"])
            actual_ts = inv["post_ts"] or clean_ts
            inv["_rows"] = {}
            for role, account, debit, credit in (("ar", "AR", inv["total"], 0), ("rev", "REVENUE", 0, inv["total"])):
                inv["_rows"][role] = add(self.rows, actual_ts, account, debit, credit, inv["currency"], inv["entity"], inv["key"], None)
                if not inv.get("dup_of"):
                    add(self.clean_rows, clean_ts, account, debit, credit, inv["currency"], inv["entity"], inv["key"], None)
        for pay in self.payments:
            spec, info = self._payment_rows(pay, apply_plan=True)
            pay["_rows"] = {}
            for role, account, debit, credit, cur, inv_key in spec:
                pay["_rows"][role] = add(self.rows, pay["ts"], account, debit, credit, cur, pay["entity"], inv_key, pay["key"])
            pay["_info"] = info
            for role, account, debit, credit, cur, inv_key in self._payment_rows(pay, apply_plan=False)[0]:
                add(self.clean_rows, pay["ts"], account, debit, credit, cur, pay["entity"], inv_key, pay["key"])
        for entry in self.journal:
            for target in (self.rows, self.clean_rows):
                add(target, entry["ts"], "EXPENSE", entry["amount"], 0, entry["currency"], None, None, None)
                add(target, entry["ts"], "CASH", 0, entry["amount"], entry["currency"], None, None, None)

    # ---- эталон
    def build_truth(self):
        issues = []
        for plant in self.plants:
            kind = plant["kind"]
            if kind == "duplicate_invoice":
                orig, dup = self.inv_by_key[plant["orig"]], self.inv_by_key[plant["dup"]]
                issues.append({"type": kind, "affected": [orig["key"], dup["key"]], "expected": orig["total"],
                               "currency": orig["currency"], "evidence": [[orig["doc"], dup["doc"]]], "plant": plant})
            elif kind == "period_cutoff":
                inv = self.inv_by_key[plant["inv"]]
                issues.append({"type": kind, "affected": [inv["key"]], "expected": inv["total"], "currency": inv["currency"],
                               "evidence": [[inv["doc"], inv["_rows"]["ar"]], [inv["doc"], inv["_rows"]["rev"]]], "plant": plant})
            else:
                pay = self.pay_by_key[plant["pay"]]
                info, rows = pay["_info"], pay["_rows"]
                if plant["alloc"]:
                    issues.append({"type": "payment_allocation", "affected": [pay["key"]], "expected": info["unapplied"],
                                   "currency": info["leg"], "evidence": [[pay["doc"], rows["un"]]], "plant": plant})
                if plant["fx"]:
                    legs = [rows[r] for r in ("ar", "un") if r in rows]
                    issues.append({"type": "fx_error", "affected": [pay["key"]], "expected": abs(info["correct"] - info["booked"]),
                                   "currency": info["leg"], "evidence": [[pay["doc"], *legs, info["fx_key"]]], "plant": plant})
        return issues

    # ---- выгрузка (внешние ID, порядок)
    def export(self) -> Episode:
        p = self.p
        ext = {}

        def assign(prefixes, keys, low, high):
            prefix = p.choice(prefixes)
            for key, num in zip(keys, p.sample(range(low, high), len(keys))):
                ext[key] = f"{prefix}-{num}"

        assign(["CUST", "C", "CL"], list(self.entities), 1000, 9999)
        assign(["INV", "SI", "BILL"], [i["key"] for i in self.invoices], 10000, 99999)
        assign(["DOC", "D"], [i["doc"] for i in self.invoices] + [x["doc"] for x in self.payments], 100000, 999999)
        assign(["PAY", "RCPT", "P"], [x["key"] for x in self.payments], 10000, 99999)
        assign(["JE", "L", "TX"], [r["key"] for r in self.rows], 1000000, 9999999)
        e = lambda key: None if key is None else ext.get(key, key)  # fx-ключи остаются как есть

        def stamp(ts):
            offset = self.company_tz if p.random() < 0.5 else timezone(timedelta(hours=p.choice(DOC_OFFSETS)))
            return ts.astimezone(offset).isoformat()

        def ordered(items, time_key):
            """По времени (ничьи случайно) или, при variant.shuffle, в случайном порядке."""
            for it in items:
                it["_tb"] = p.random()
            if self.cfg["variant"]["shuffle"]:
                return sorted(items, key=lambda it: it["_tb"])
            return sorted(items, key=lambda it: (it[time_key], it["_tb"]))

        ledger = [{"row_id": e(r["key"]), "timestamp": stamp(r["ts"]), "account": r["account"],
                   "debit_minor": r["debit"], "credit_minor": r["credit"], "currency": r["currency"],
                   "entity_id": e(r["entity"]), "invoice_id": e(r["invoice"]), "payment_id": e(r["payment"])}
                  for r in ordered(self.rows, "ts")]

        invoices = []
        for inv in ordered(self.invoices, "issued"):
            off = self.company_tz if p.random() < 0.5 else timezone(timedelta(hours=p.choice(DOC_OFFSETS)))
            invoices.append({"doc_id": e(inv["doc"]), "invoice_id": e(inv["key"]), "entity_id": e(inv["entity"]),
                             "issued_at": inv["issued"].astimezone(off).isoformat(),
                             "due_at": (inv["issued"] + timedelta(days=30)).astimezone(off).isoformat(),
                             "currency": inv["currency"], "total_minor": inv["total"], "line_items": inv["lines"]})
        payments = [{"doc_id": e(x["doc"]), "payment_id": e(x["key"]), "timestamp": stamp(x["ts"]),
                     "currency": x["currency"], "amount_minor": x["amount"],
                     "invoice_refs": [{"invoice_id": e(k), "amount_minor": a} for k, a in x["refs"]]}
                    for x in ordered(self.payments, "ts")]
        fx = [{"date": d, "source_currency": c, "target_currency": self.base, "rate": rate}
              for (c, d), rate in sorted(self.fx.items(), key=lambda kv: (kv[0][1], kv[0][0]))]

        manifest = {
            "schema_version": SCHEMA_VERSION, "episode_id": self.episode_id, "base_currency": self.base,
            "period_start": self.start.astimezone(self.company_tz).isoformat(),
            "period_end": self.end.astimezone(self.company_tz).isoformat(),
            "timezone": tu.offset_text(self.company_tz),
            "currency_exponents": {c: money.EXPONENT[c] for c in sorted({self.base, *self.foreign})},
            "counts": {"ledger_rows": len(ledger), "documents": len(invoices) + len(payments), "invoices": len(invoices),
                       "payments": len(payments), "fx_rows": len(fx)},
            "rounding": {"mode": self.rounding, "unit": "minor units of the target currency"},
            "files": ["ledger.csv", "invoices.jsonl", "payments.jsonl", "fx.csv", "policy.json"],
        }
        policy = {
            "fx_rate_date": self.fx_rule,
            "fx_rule": ("Платёж в валюте, отличной от валюты счёта, пересчитывается по курсу fx.csv "
                        "(source_currency = валюта платежа, target_currency = базовая) на дату платежа: "
                        + ("дата по часовому поясу компании." if self.fx_rule == "company_date" else "дата в UTC.")),
            "period_rule": "Период [period_start, period_end): сравнивайте мгновения, а не строки. "
                           "Счёт и его проводки должны попадать в один и тот же период.",
            "posting_rule": "Счёт: дебет AR и кредит REVENUE на total_minor. Платёж: дебет CASH на amount_minor, "
                            "кредит AR по каждой ссылке invoice_refs (в валюте счёта). Остаток уходит в UNAPPLIED.",
            "unapplied_rule": "UNAPPLIED законен только для платежа без invoice_refs (аванс).",
            "posting_time_rule": "Проводку счёта могут сделать позже или раньше даты счёта. Ошибка только тогда, "
                                 "когда счёт и его проводка попадают в разные периоды.",
            "partial_payment_rule": "Счёт можно оплачивать частями, в том числе в другой валюте: остаток AR после "
                                    "частичной оплаты не ошибка.",
            "rounding_residual_rule": "При оплате в другой валюте возможен остаток или переплата AR из-за округления "
                                      "до минимальной единицы валюты платежа; это не ошибка.",
            "duplicate_rule": "Дубль: тот же клиент, валюта, сумма и тот же состав line_items "
                              "(включая период услуги) в разных счетах.",
            "rounding": self.rounding,
            "null_rule": "Пустая ссылка в CSV пишется как null; связи не додумываются.",
        }

        issues = []
        for issue in self.build_truth():
            affected = sorted(e(k) for k in issue["affected"])
            issues.append({"type": issue["type"], "affected_ids": affected,
                           "expected_amount_minor": issue["expected"], "currency": issue["currency"],
                           "tolerance_minor": TOLERANCE[issue["type"]],
                           "evidence_sets": [sorted(e(k) for k in s) for s in issue["evidence"]],
                           "_plant": issue["plant"]})
        issues.sort(key=lambda i: (i["type"], i["affected_ids"]))
        plants = [i.pop("_plant") for i in issues]
        for n, issue in enumerate(issues, 1):
            issue["issue_id"] = f"ISS-{n:02d}"
        truth = {"schema_version": SCHEMA_VERSION, "episode_id": self.episode_id, "issues": issues}

        changes = []
        for issue, plant in zip(issues, plants):
            detail = {"issue_id": issue["issue_id"], "kind": plant["kind"], "interaction": plant.get("inter")}
            if plant["kind"] == "payment_plan":
                pay = self.pay_by_key[plant["pay"]]
                plan, info = pay["plan"], pay["_info"]
                detail.update({"payment_id": e(pay["key"]), "plan": {k: v for k, v in plan.items()},
                               "correct_credit_minor": info["correct"], "booked_total_minor": info["booked"],
                               "unapplied_minor": info["unapplied"]})
            elif plant["kind"] == "period_cutoff":
                detail["variant"] = plant.get("variant")
            changes.append(detail)
        clean = [{"timestamp": r["ts"].isoformat(), "account": r["account"], "debit_minor": r["debit"],
                  "credit_minor": r["credit"], "currency": r["currency"], "entity_id": e(r["entity"]),
                  "invoice_id": e(r["invoice"]), "payment_id": e(r["payment"])} for r in self.clean_rows]
        provenance = {
            "generator_version": GEN_VERSION, "seed": self.seed, "config": self.cfg, "episode_id": self.episode_id,
            "history": {"entities": len(self.entities), "foreign_currencies": self.foreign,
                        "distractor_tags": sorted({x["tag"] for x in self.invoices + self.payments if x["tag"] != "ordinary"})},
            "changes": changes, "clean_ledger": clean,
        }
        return Episode(episode_id=self.episode_id, manifest=manifest, policy=policy, ledger=ledger,
                       invoices=invoices, payments=payments, fx=fx, truth=truth, provenance=provenance)


def generate(seed: int, config: dict | None = None) -> Episode:
    """Эпизод по seed и config. Тот же seed + config + версия = те же данные байт в байт."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    cfg = merge_config(config)
    with localcontext(Context(prec=50, rounding=ROUND_HALF_EVEN)):  # результат не зависит от глобального decimal
        builder = _Builder(seed, cfg)
        builder.build_ordinary()
        builder.build_journal()
        builder.build_distractors()
        builder.plant_errors()
        builder.post_all()
        episode = builder.export()
    documents = len(episode.invoices) + len(episode.payments)
    if not MIN_LEDGER <= len(episode.ledger) <= MAX_LEDGER or documents > MAX_DOCUMENTS:
        raise ValueError(f"config gives {len(episode.ledger)} ledger rows and {documents} documents; "
                         f"allowed: {MIN_LEDGER}..{MAX_LEDGER} rows, at most {MAX_DOCUMENTS} documents")
    return episode
