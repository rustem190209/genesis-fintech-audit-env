# AuditEnv: среда для ИИ-аудитора (GENESIS, трек FinTech)

## Что это

AuditEnv — учебная среда (в стиле обучения с подкреплением, RL), в которой ИИ-агент играет роль аудитора.

Агенту дают бухгалтерские данные одной компании за один отчётный период: журнал проводок, счета, платежи, курсы валют и правила учёта. В данных могут быть спрятаны ошибки четырёх типов (а могут и не быть). Агент читает данные через несколько инструментов, а в конце один раз сдаёт список найденных ошибок. За это он получает награду от -1 до 1.

Четыре типа ошибок:

- `duplicate_invoice` — дубль счёта;
- `payment_allocation` — часть платежа лежит на счёте UNAPPLIED, хотя должна быть на AR;
- `fx_error` — неверный курс при пересчёте платежа в иностранной валюте;
- `period_cutoff` — счёт и его проводки попали в разные периоды.

Все данные генерируются программой по числу `seed` (зерно случайности). Одинаковый seed и config дают одинаковые данные байт в байт. Подробности устройства в [DESIGN.md](DESIGN.md).

## Требования

- Python 3.11 или новее.
- Сама среда использует только стандартную библиотеку Python.
- `pytest` нужен только для тестов.

Установка pytest (PowerShell):

```powershell
cd C:\Users\User\Projects\hackathon
pip install -r requirements-dev.txt
```

## Быстрая проверка одной командой

```powershell
python check.py
```

Команда делает три шага:

1. Запускает тесты (`pytest`).
2. Проходит по 20 открытым эпизодам (10 train и 10 dev): проверяет данные и эталон, а также что повторная генерация даёт те же данные.
3. Прогоняет 4 базовые политики и проверяет порядок их наград: `oracle` > `simple_reconciler` > `random_explorer` и `flag_all`.

Сеть, платные API и GPU не нужны. Код возврата 0 значит, что всё в порядке. В конце печатается `ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ`. Если что-то не так, печатается `ПРОВЕРКА НЕ ПРОЙДЕНА` и код возврата 1.

Проверить код возврата в PowerShell:

```powershell
python check.py; echo $LASTEXITCODE
```

## Запуск в контейнере (Docker)

Команды взяты из `Dockerfile`:

```powershell
docker build -t audit-env .
docker run --rm --network none --cpus 2 --memory 4g audit-env
```

Сборка требует сеть (ставится pytest). Сам запуск идёт без сети и выполняет `python check.py`.

Важно: на нашей машине Docker не установлен, поэтому контейнер локально мы не запускали и не проверяли. Команды написаны по `Dockerfile`, но результат не подтверждён.

## Как сгенерировать эпизоды и траектории

Эпизоды (10 train + 10 dev) попадают в папку `episodes\`:

```powershell
python scripts\generate_episodes.py
```

Каждый эпизод — это папка с открытыми файлами и подпапкой `evaluator\` с закрытым эталоном (`truth.json`, `provenance.json`). Флаг `--no-closed` не пишет закрытые файлы. Флаг `--out ПАПКА` меняет папку результата. Кроме эпизодов, создаётся `episodes\index.json`.

Траектории базовых политик (запись всех шагов) и таблица результатов:

```powershell
python scripts\run_baselines.py
```

Результат в папке `results\`:

- `trajectories.jsonl` — все шаги всех политик на всех 20 эпизодах;
- `baseline_results.md` — таблица средних метрик;
- `baseline_results.json` — те же результаты подробнее, по каждому эпизоду.

## Как пользоваться средой (пример на Python)

Запускайте из корня проекта. Примеры ниже вставьте в файл, например `example.py`, и запустите `python example.py`.

```python
from audit_env.env import AuditEnv

env = AuditEnv(mode="open")
obs, info = env.reset(seed=1)          # создать эпизод по seed
print(obs["task"])                      # текст задания для агента
print(obs["tables"])                    # сколько строк в каждой таблице

# посмотреть первые 5 счетов
obs, reward, terminated, truncated, info = env.step(
    {"tool": "list_records", "args": {"table": "invoices", "limit": 5}}
)
print(obs["result"]["records"])

# сдать ответ: пустой список (считаем, что ошибок нет)
obs, reward, terminated, truncated, info = env.step(
    {"tool": "submit",
     "args": {"findings": {"episode_id": info["episode_id"], "findings": []}}}
)
print(terminated, reward)
```

Что здесь есть:

- `reset(seed, config)` возвращает наблюдение и `info` с `episode_id`. `config` необязателен (описание ключей в `audit_env\generator.py`, функция `default_config`).
- `step(action)` возвращает `(observation, reward, terminated, truncated, info)`. Действие — словарь `{"tool": имя, "args": {...}}`.
- Инструменты: `list_records`, `get_record`, `get_document`, `calculator`, `submit`. Каждый вызов стоит один шаг.
- Лимит: 100 шагов (включая `submit`) и 60 секунд на эпизод.
- `submit` завершает эпизод. Формат вывода (finding) описан в тексте задания `obs["task"]`.

Два режима:

| Режим    | Что получает политика              | Для чего                                                            |
| -------- | ---------------------------------- | ------------------------------------------------------------------- |
| `open`   | итоговую награду на последнем шаге | обучение                                                            |
| `closed` | на всех шагах награда 0.0          | проверка; итог читает только проверяющий через `env.final_report()` |

Пример готового прогона политики (из `audit_env\runner.py`):

```python
from audit_env.runner import run_episode

report = run_episode("simple_reconciler", seed=101, mode="open")
print(report["reward"], report["precision"], report["recall"])
```

Имена политик: `random_explorer`, `simple_reconciler`, `flag_all`, `oracle`.

## Запуск среды отдельным процессом

Среда умеет работать как отдельный процесс: общение идёт JSON-строками через stdin и stdout. Так оценщик и эталон остаются вне процесса агента.

```powershell
python -m audit_env.server --mode closed
```

Параметры: `--mode open|closed` (по умолчанию `open`) и `--harness`. Команды (по одной JSON-строке):

- `{"cmd": "reset", "seed": 1, "config": {}}`
- `{"cmd": "step", "action": {"tool": "calculator", "args": {"expression": "2*21"}}}`
- `{"cmd": "report"}` — итог эпизода; работает только если сервер запущен с флагом `--harness` (это для проверяющего, агенту отчёт не отдаётся).

При ошибке команды сервер отвечает `{"error": "..."}` и не падает.

Про изоляцию. Методы `final_report()` и `evaluator_truth()` в классе `AuditEnv` нужны проверяющему. Если агент работает в том же процессе и держит объект `env`, он до них дотянется. Настоящая изоляция даёт только серверный режим: агент видит лишь ответы на команды, а `report` без `--harness` не работает.

Секрет проверяющего. Если задать переменную окружения `AUDIT_ENV_SECRET`, номер эпизода (`episode_id`) перестанет зависеть только от seed и config, и агент не сможет подобрать seed перебором. Сами данные эпизода от секрета не меняются.

```powershell
$env:AUDIT_ENV_SECRET = "[ЗАПОЛНИТЬ: любая секретная строка]"
python -m audit_env.server --mode closed
```

Для проверяющего:

```powershell
python -m audit_env.server --mode closed --harness
```

## Структура репозитория

| Путь                                                             | Что там                                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `audit_env\generator.py`                                         | генератор эпизодов (скрытая история, отвлекающие случаи, ошибки, эталон) |
| `audit_env\env.py`                                               | класс `AuditEnv`: `reset`, `step`, инструменты, лимиты                   |
| `audit_env\server.py`                                            | среда как отдельный процесс (JSON-строки)                                |
| `audit_env\grader\matcher.py`                                    | проверка формата выводов и сопоставление с эталоном                      |
| `audit_env\grader\reward.py`                                     | P, R, F1 и итоговая награда                                              |
| `audit_env\grader\evaluator.py`                                  | оценщик (держит эталон)                                                  |
| `audit_env\grader\validate.py`                                   | проверки данных эпизода и независимый детектор                           |
| `audit_env\policies.py`                                          | 4 базовые политики                                                       |
| `audit_env\runner.py`                                            | прогон политики через среду с записью траектории                         |
| `audit_env\plan.py`                                              | план 20 открытых эпизодов (seed и параметры)                             |
| `audit_env\files.py`, `money.py`, `timeutil.py`, `calculator.py` | запись файлов, деньги и округление, время и часовые пояса, калькулятор   |
| `scripts\generate_episodes.py`                                   | создаёт `episodes\`                                                      |
| `scripts\run_baselines.py`                                       | создаёт `results\`                                                       |
| `check.py`                                                       | единая команда проверки                                                  |
| `tests\`                                                         | тесты (pytest)                                                           |
| `episodes\`                                                      | 10 train и 10 dev эпизодов                                               |
| `results\`                                                       | траектории и таблица базовых политик                                     |
| `Dockerfile`, `.dockerignore`, `requirements-dev.txt`            | запуск в контейнере и зависимости для тестов                             |
| `LICENSE`                                                        | лицензия MIT                                                             |
| `README.md`, `DESIGN.md`                                         | эта инструкция и записка о дизайне                                       |

## Что сдаём

- Исходники с лицензией (`audit_env\`, `scripts\`, `check.py`, `LICENSE`).
- README (этот файл).
- Matcher и функция награды (`audit_env\grader\matcher.py`, `audit_env\grader\reward.py`).
- 10 train и 10 dev эпизодов в `episodes\`.
- Тесты (`tests\`).
- 4 базовые политики (`audit_env\policies.py`).
- Траектории `results\trajectories.jsonl` и таблица `results\baseline_results.md`.
- Записка о дизайне `DESIGN.md`.
- Демо: [ЗАПОЛНИТЬ: ссылка на демо-видео].
- Репозиторий: [ЗАПОЛНИТЬ: ссылка на GitHub].
