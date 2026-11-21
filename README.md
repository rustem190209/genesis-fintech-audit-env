# AuditEnv: среда для ИИ-аудитора (GENESIS, трек FinTech)

## Коротко для жюри

- **Одна команда проверки** (рабочая директория — корень репозитория, в контейнере `/app`):
  ```
  python check.py
  ```
  или в контейнере: `docker build -t audit-env .` и `docker run --rm --network none --cpus 2 --memory 4g audit-env`.
- **Подключение по «Определениям проверок»** — файл `judge_adapter.py` в корне: `generate_episode`, `make_env`, `score_episode` (раздел [Подключение жюри](#подключение-жюри-judge_adapterpy)).
- **Где что проверяется** — раздел [Карта критериев](#карта-критериев).
- Python 3.11+, только стандартная библиотека; `pytest` нужен только для тестов. Сеть, платные API и GPU не нужны.

## Что это

AuditEnv — среда для обучения с подкреплением (RL), в которой ИИ-агент играет роль аудитора. Агенту дают бухгалтерские данные одной вымышленной компании за один отчётный период: журнал проводок, счета, платежи, курсы валют и правила учёта. В данных могут быть спрятаны ошибки четырёх типов, а могут и не быть. Агент читает данные инструментами, один раз сдаёт список найденных ошибок и получает награду от -1 до 1.

| Тип (`type`)         | Что это                                                                      | Канонические `affected_ids`                |
| -------------------- | ---------------------------------------------------------------------------- | ------------------------------------------ |
| `duplicate_invoice`  | дубль счёта: тот же клиент, валюта, сумма и состав строк                     | `{invoice_id оригинала, invoice_id дубля}` |
| `payment_allocation` | часть платежа (или весь) в UNAPPLIED вместо AR                               | `{payment_id}`                             |
| `fx_error`           | платёж в другой валюте пересчитан по неверной дате курса или неверному курсу | `{payment_id}`                             |
| `period_cutoff`      | счёт и его проводка попали в разные периоды                                  | `{invoice_id}`                             |

Все данные генерируются по `seed` и `config`. Одинаковые seed, config и версия дают одинаковые данные байт в байт. Устройство и ограничения — в [DESIGN.md](DESIGN.md).

## Быстрая проверка одной командой

```powershell
python check.py
```

Четыре шага:

1. Тесты (`pytest`): контракт, изоляция, генератор, matcher и награда, атаки, все таблицы из «Определений проверок».
2. 20 открытых эпизодов (10 train и 10 dev): данные и эталон, повторная генерация даёт те же байты (в том же процессе и в отдельном с другим `PYTHONHASHSEED`).
3. Четыре базовые политики и порядок их наград: `oracle` > `simple_reconciler` > `random_explorer` и `flag_all`.
4. Протокол жюри: каждый эпизод через `generate_episode` в папку, четыре политики через `make_env` в закрытом режиме с `episode_dir` и оценка `score_episode` (награды совпадают с прямым прогоном); парная копия (перестановка записей, смена ID или оба): данные корректны, oracle с новыми ID получает те же метрики.

Код возврата 0 и строка `ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ` — всё в порядке, иначе `ПРОВЕРКА НЕ ПРОЙДЕНА` и код 1.

Контейнер проверен на Docker 29.9.3 (Windows, WSL2) с `--network none --cpus 2 --memory 4g`. Эпизоды, сгенерированные внутри Linux-контейнера, побайтно совпадают с папкой `episodes\`.

## Подключение жюри (`judge_adapter.py`)

Тонкий Python-адаптер по разделу «Единый способ подключения». Политики организаторов те же; адаптер только переводит оболочку действий и форму ответов.

```python
import judge_adapter as ja

ja.generate_episode(seed, config, output_dir)          # эпизод в папку жюри
env = ja.make_env()                                     # объект с reset/step
obs, info = env.reset(seed, {"mode": "closed", "episode_dir": output_dir, "team_config": {}})
obs, reward, terminated, truncated, info = env.step({"name": "list_records", "args": {"table": "ledger", "cursor": None, "limit": 50}})
ja.score_episode(output_dir, findings, inspection_steps)   # только у жюри
```

**`generate_episode(seed, config, output_dir)`** пишет эпизод:

| Файл                        | Что внутри                                                                                                                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `manifest.json`             | schema_version, episode_id, base_currency, period_start/end (со смещением), timezone (фиксированное смещение, например `+05:00`), counts (ledger_rows, documents, invoices, payments, fx_rows), rounding, currency_exponents |
| `policy.json`               | правило даты курса (`fx_rate_date`: `company_date` или `utc_date`), период, проводки, UNAPPLIED, дубли, частичные оплаты, время проводки, остатки округления                                                                 |
| `ledger.csv`                | row_id, timestamp, account, debit_minor, credit_minor, currency, entity_id, invoice_id, payment_id; пустая ссылка — `null`                                                                                                   |
| `invoices.jsonl`            | doc_id, invoice_id, entity_id, issued_at, due_at, currency, total_minor, line_items                                                                                                                                          |
| `payments.jsonl`            | doc_id, payment_id, timestamp, currency, amount_minor, invoice_refs `[{invoice_id, amount_minor}]`                                                                                                                           |
| `fx.csv`                    | date, source_currency, target_currency, rate (десятичная строка)                                                                                                                                                             |
| `evaluator/truth.json`      | закрыто: `issues[]` с issue_id, type, affected_ids, expected_amount_minor, currency, tolerance_minor, evidence_sets                                                                                                          |
| `evaluator/provenance.json` | закрыто: версия генератора, seed, config, внесённые изменения (`changes`) и «чистый» журнал до ошибок (`clean_ledger`)                                                                                                       |

`config` — config генератора (или объект с полем `team_config`). Ключи и границы — `default_config()` в `audit_env/generator.py`:

- `errors` — число ошибок каждого типа (каждый тип настраивается отдельно, всего до 12);
- `interactions` — сценарии взаимодействия: `fx_boundary_partial`, `dup_cutoff`;
- `error_modes` — явный вариант ошибки:
  - `payment_allocation`: `missing` или `partial`;
  - `fx_error`: `wrong_date` или `wrong_rate`;
  - `period_cutoff`: `end_late`, `end_early`, `start_late` или `start_early`;
  - по умолчанию для всех — `random`;
- `distractors` — число корректных отвлекающих случаев каждого вида (0–10);
- `n_invoices` (20–80), `n_journal` (0–800), `base_currency`;
- `variant` — другие ID и порядок при той же истории.

**`reset(seed, config)`** с `config = {"mode": "closed"|"train", "episode_dir": ..., "team_config": {...}}`:

- с `episode_dir` загружает именно этот эпизод, не генерирует другой и не меняет ID; повторный reset начинает его заново;
- в `closed` закрытые файлы не читаются вовсе;
- без `episode_dir` эпизод генерируется по seed и `team_config`;
- первое наблюдение: `{"task", "manifest", "policy", "remaining_steps": 100}`.

**`step({"name": ..., "args": {...}})`** → `({"result", "error", "remaining_steps"}, reward, terminated, truncated, info)`:

| Действие       | args                                                                                | result                                               |
| -------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `list_records` | `table` (ledger, invoices, payments, fx), `cursor` (первый — null), `limit` (до 50) | `{"records": [...], "next_cursor": строка или null}` |
| `get_record`   | `table`, `id` (row_id, invoice_id, payment_id или `ДАТА:SRC:DST` для fx)            | объект записи (у счёта — с line_items)               |
| `get_document` | `doc_id`                                                                            | объект документа (счёт или платёж)                   |
| `calculator`   | `expression` (только `+ - * / ( )` и десятичные числа ASCII)                        | десятичная строка, например `"0.3"`                  |
| `submit`       | `findings` (объект findings.json)                                                   | `{"accepted": true}` или ошибка структуры            |

- **Ошибка** — `{"code", "message"}` без путей, seed и закрытых данных. Коды: `invalid_action`, `unknown_tool`, `invalid_args`, `invalid_cursor`, `not_found`, `calc_error`, `invalid_submission`, `time_limit`, `episode_over`. Любая ошибка расходует шаг.
- **В `info`** только `episode_id` и `step`.
- **Награда.** В `closed` reward всегда 0, в том числе на завершении. В `train` политике возвращается терминальная награда.
- **Лимиты.** 100 действий, submit входит в них, но не в `inspection_steps`. 60 секунд от начала reset до конца, включая время политики. При превышении: `truncated=true`, итоговый ответ пустой. После завершения новые действия ничего не меняют (ошибка `episode_over`).

**`score_episode(episode_dir, findings, inspection_steps)`** → `{"tp", "fp", "fn", "precision", "recall", "f1", "return"}`. Невалидный submit: `return = -1`, остальные поля `null`, ошибка в журнале оценщика (logger `audit_env.evaluator`).

**Правила сопоставления** (`audit_env/grader/matcher.py`, точно по определениям):

- **ID, `type` и `currency`** сравниваются точно: регистр и пробелы важны.
- **`affected_ids`** — множество, должно точно совпасть с каноническим набором из таблицы выше.
- **`evidence_ids`** — множество после проверки лимита 10 (вместе с повторами):
  - нужен хотя бы один полный допустимый набор из эталона;
  - лишние существующие ID допустимы;
  - несуществующий ID делает вывод неверным (FP).

  Допустимые ID доказательств: row_id, invoice_id, payment_id, doc_id и `ДАТА:SRC:DST`.

- **Сумма** в минимальных единицах, допуск **τ = 0** для всех четырёх типов: суммы считаются точно по правилу округления из `policy.json`.
- **Неверная структура → `return = -1`.** Это 31 вывод, 11 ID доказательств, explanation длиннее 1000 символов, неверный тип значения или отсутствующее поле.
- **Структурно верный, но неправильный вывод — FP.** Лишние ключи игнорируются.
- **Сопоставление** — паросочетание максимальной мощности. Среди равных выбирается детерминированно по issue_id и индексу вывода. Порядок findings не меняет TP, FP, FN и награду.

**Изоляция.**

- В закрытом режиме с `episode_dir` процесс среды читает только открытые файлы.
- Эталон читает `score_episode` у жюри. Политике остаются только наблюдения.
- Внутри одного Python-процесса скрытие полей изоляцией не считается. Поэтому жюри запускает политику отдельным процессом, а у нас для этого есть режим агента сервера (ниже).

## Карта критериев

| Критерий (раздел 10 кейса)        | Где закрыто                                                                                                                                                                                                 |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Контракт и изоляция, 25           | `judge_adapter.py`; `tests/test_judge_spec.py` (C01–C20), `tests/test_contract.py`, `tests/test_isolation.py`, `tests/test_hardening.py`                                                                    |
| Данные и эталон, 25               | `audit_env/generator.py` (история → ошибки → эталон), `audit_env/grader/validate.py` (баланс до ошибки, расхождение после, независимый детектор, доказательства), `scripts/stress_check.py` (1800 эпизодов) |
| Надёжность награды, 25            | `audit_env/grader/matcher.py`, `reward.py`; `tests/test_judge_spec.py` (20 тестов блока 3 с теми же числами, M01–M12), `tests/test_reward.py`, `tests/test_shortcuts.py`                                    |
| Воспроизводимость и обобщение, 15 | `audit_env/transform.py` (перестановка и смена ID, как у жюри), шаг 4 в `check.py`, `tests/test_judge_spec.py` (блок 4), проверка байтов в отдельном процессе                                               |
| Инженерия, 10                     | одна команда `check.py`; версии: `GEN_VERSION`, `schema_version`, `requirements-dev.txt`, seed и config в `episodes/index.json`; `results/trajectories.jsonl`; разбор сбоев и ограничения в `DESIGN.md`     |

## Как сгенерировать эпизоды и траектории

```powershell
python scripts\generate_episodes.py     # 10 train + 10 dev в episodes\ (+ episodes\index.json с seed и config)
python scripts\run_baselines.py         # results\trajectories.jsonl, baseline_results.md, baseline_results.json
python scripts\stress_check.py          # 300 seed × 6 config = 1800 эпизодов, все проверки данных
```

`--no-closed` у генерации не пишет закрытые файлы, `--out ПАПКА` меняет папку. В `baseline_results.json` поля `wall_s` и `elapsed_s` — время, они меняются от запуска к запуску.

## Наш собственный интерфейс (без адаптера)

```python
from audit_env.env import AuditEnv

env = AuditEnv(mode="open")
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step({"tool": "list_records", "args": {"table": "invoices", "limit": 5}})
obs, reward, terminated, truncated, info = env.step(
    {"tool": "submit", "args": {"findings": {"episode_id": info["episode_id"], "findings": []}}})
```

Здесь действие — `{"tool", "args"}`, наблюдение содержит `budget_remaining`, а результаты завёрнуты (`{"record": ...}`, `{"value": ...}`). Адаптер жюри приводит всё к формату определений. Режимы: `open` (терминальная награда политике) и `closed` (награда 0, итог читает проверяющий через `env.final_report()`). Прогон политики: `audit_env.runner.run_episode("simple_reconciler", seed=101)`; через адаптер: `judge_adapter.run_policy(name, episode_dir)`.

## Среда отдельным процессом

Режим нужно выбрать явно.

**Режим агента (изолированный).** Эпизод выбирает проверяющий при запуске. Агенту доступны только `{"cmd": "start"}` и `{"cmd": "step", "action": {...}}`, а `reset` и `report` запрещены. Итог пишется в файл проверяющего:

```powershell
python -m audit_env.server --seed 42 --config config.json --mode closed --report report.json
```

**Доверенный режим** (`--harness`, только для проверяющего): доступны `reset`, `step`, `report`. При ошибке команды сервер отвечает `{"error": "..."}` (не длиннее 200 символов) и не падает.

**Секрет проверяющего.** Генератор открыт, поэтому, зная config, можно перебрать seed и восстановить эпизод. Переменная окружения `AUDIT_ENV_SECRET` входит в ключ всех случайных чисел. С ней получаются другие компании, а без секрета их не воспроизвести. Одинаковые seed, config и секрет дают одинаковые данные.

## Структура репозитория

| Путь                                                             | Что там                                                                 |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `judge_adapter.py`                                               | адаптер жюри: generate_episode, make_env, score_episode, run_policy     |
| `audit_env\generator.py`                                         | генератор (скрытая история, отвлекающие случаи, ошибки, эталон)         |
| `audit_env\env.py`                                               | `AuditEnv`: reset, step, инструменты, лимиты, загрузка эпизода из папки |
| `audit_env\grader\matcher.py`, `reward.py`, `evaluator.py`       | структура выводов, сопоставление, награда, оценщик                      |
| `audit_env\grader\validate.py`                                   | проверки данных и независимый детектор                                  |
| `audit_env\transform.py`                                         | перестановка записей и смена ID в копии эпизода (парные проверки)       |
| `audit_env\policies.py`, `runner.py`                             | 4 базовые политики и прогон с траекторией                               |
| `audit_env\server.py`                                            | среда отдельным процессом                                               |
| `audit_env\plan.py`                                              | план 20 открытых эпизодов (seed и config)                               |
| `audit_env\files.py`, `money.py`, `timeutil.py`, `calculator.py` | файлы эпизода, деньги, время, калькулятор                               |
| `scripts\`                                                       | генерация эпизодов, траектории, стресс-проверка                         |
| `check.py`                                                       | единая команда проверки                                                 |
| `tests\`                                                         | тесты (pytest)                                                          |
| `episodes\`, `results\`                                          | 20 эпизодов; траектории и таблица базовых политик                       |
| `Dockerfile`, `requirements-dev.txt`, `LICENSE`                  | контейнер, зафиксированные зависимости, лицензия MIT                    |

## Что сдаём

- Исходники с лицензией и зафиксированными зависимостями (`audit_env\`, `judge_adapter.py`, `scripts\`, `check.py`, `LICENSE`, `requirements-dev.txt`).
- README (этот файл) и записка о дизайне `DESIGN.md`.
- Matcher и награда: `audit_env\grader\matcher.py`, `reward.py`.
- 10 train и 10 dev эпизодов: `episodes\`.
- Тесты: `tests\`.
- 4 базовые политики: `audit_env\policies.py`; они же через адаптер жюри: `judge_adapter.run_policy`.
- Траектории: `results\trajectories.jsonl`, таблица: `results\baseline_results.md`.
- Демо: [ЗАПОЛНИТЬ: ссылка на демо-видео].
- Репозиторий: https://github.com/rustem190209/genesis-fintech-audit-env.
