# Автономная проверка: Python 3.11, без сети во время запуска, без GPU.
# Сборка (нужна сеть, один раз):  docker build -t audit-env .
# Проверка (сеть не нужна):       docker run --rm --network none --cpus 2 --memory 4g audit-env
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

CMD ["python", "check.py"]
