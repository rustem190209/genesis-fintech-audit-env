"""Среда как отдельный процесс: JSON-строки через stdin/stdout.

Два режима, выбор обязателен:

1. Режим агента (изолированный). Эпизод выбирает проверяющий при запуске, агент его сменить не может:
     python -m audit_env.server --seed 42 [--config cfg.json] [--mode closed] [--report report.json]
   Агенту доступны только команды:
     {"cmd": "start"}                 -> {"observation": ..., "info": ...}  (первое наблюдение; можно повторять)
     {"cmd": "step", "action": {...}} -> {"observation", "reward", "terminated", "truncated", "info"}
   reset и report агенту запрещены. Итог эпизода сервер пишет в файл --report (его читает только
   проверяющий); агенту он не отдаётся. Эталон и оценщик живут в процессе сервера, а не агента.

2. Доверенный режим (--harness): клиент — сам проверяющий. Доступны reset, step и report:
     {"cmd": "reset", "seed": 1, "config": {...}} / {"cmd": "step", ...} / {"cmd": "report"}
   Этот режим нельзя отдавать агенту: через reset он мог бы выбрать себе другой эпизод.

Ошибка команды -> {"error": "..."}; сервер не падает от чужого ввода.
"""
import argparse
import json
import sys
from pathlib import Path

from .env import AuditEnv

MAX_ERROR_LEN = 200  # чужой ввод не возвращаем целиком


def _error(text) -> dict:
    return {"error": str(text)[:MAX_ERROR_LEN]}


def _safe(call) -> dict:
    try:
        return call()
    except (RuntimeError, ValueError) as exc:
        return _error(exc)
    except Exception:  # сервер не должен падать от чужого ввода
        return _error("internal error")


def _step(env, message):
    obs, reward, terminated, truncated, info = env.step(message.get("action"))
    return {"observation": obs, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}


def handle(env: AuditEnv, message, harness: bool) -> dict:
    """Доверенный режим: reset, step, report (report только при harness=True)."""
    if not isinstance(message, dict) or "cmd" not in message:
        return _error("expected an object with cmd")
    cmd = message["cmd"]
    if cmd == "reset":
        if not harness:
            return _error("reset is controlled by the harness")
        return _safe(lambda: dict(zip(("observation", "info"), env.reset(message.get("seed", 0), message.get("config")))))
    if cmd == "step":
        return _safe(lambda: _step(env, message))
    if cmd == "report":
        if not harness:
            return _error("report is not available to agents")
        return _safe(lambda: {"report": env.final_report()})
    return _error("unknown cmd")


class AgentSession:
    """Режим агента: эпизод зафиксирован при запуске, агенту доступны только start и step."""

    def __init__(self, env: AuditEnv, seed: int, config, report_path):
        self.env = env
        self.report_path = Path(report_path) if report_path else None
        self.report_written = False
        obs, info = env.reset(seed, config)
        self.first = {"observation": obs, "info": info}

    def handle(self, message) -> dict:
        if not isinstance(message, dict) or "cmd" not in message:
            return _error("expected an object with cmd")
        cmd = message["cmd"]
        if cmd == "start":
            return self.first
        if cmd == "step":
            reply = _safe(lambda: _step(self.env, message))
            self._write_report_once()
            return reply
        if cmd in ("reset", "report"):
            return _error(f"{cmd} is controlled by the harness")
        return _error("unknown cmd")

    def _write_report_once(self):
        """Итог пишется в файл проверяющего ровно один раз, когда эпизод завершён."""
        if self.report_written or self.report_path is None or not self.env._state["done"]:
            return
        self.report_path.write_text(json.dumps(self.env.final_report(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.report_written = True


def _serve(responder) -> None:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # чужие байты не должны ронять чтение
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except (ValueError, RecursionError):  # битый JSON, слишком длинное число, глубокая вложенность
            reply = _error("invalid JSON")
        else:
            reply = responder(message)
        sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="audit_env JSON-lines server")
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--harness", action="store_true", help="доверенный режим: клиент — проверяющий (reset, report)")
    parser.add_argument("--seed", type=int, help="режим агента: seed эпизода, который выбирает проверяющий")
    parser.add_argument("--config", help="режим агента: путь к JSON-файлу с config эпизода")
    parser.add_argument("--report", help="режим агента: куда записать итог эпизода для проверяющего")
    args = parser.parse_args(argv)
    if args.harness == (args.seed is not None):
        parser.error("выберите режим: --seed N (режим агента) или --harness (доверенный режим)")
    env = AuditEnv(mode=args.mode)
    if args.harness:
        _serve(lambda message: handle(env, message, harness=True))
        return 0
    config = None
    if args.config:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parser.error(f"не удалось прочитать config: {exc}")
    try:
        session = AgentSession(env, args.seed, config, args.report)
    except ValueError as exc:
        parser.error(f"неверный config: {exc}")
    _serve(session.handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
