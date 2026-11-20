"""Среда как отдельный процесс: JSON-строки через stdin/stdout.

Так оценщик и эталон живут вне процесса агента: агент видит только ответы на команды.

  python -m audit_env.server [--mode open|closed] [--harness]

Команды (по одной JSON-строке):
  {"cmd": "reset", "seed": 1, "config": {...}}  -> {"observation": ..., "info": ...}
  {"cmd": "step", "action": {...}}              -> {"observation", "reward", "terminated", "truncated", "info"}
  {"cmd": "report"}                              -> итог эпизода; только с флагом --harness (для проверяющего)
Ошибка команды -> {"error": "..."}; сервер не падает.
"""
import argparse
import json
import sys

from .env import AuditEnv


def handle(env: AuditEnv, message, harness: bool) -> dict:
    if not isinstance(message, dict) or "cmd" not in message:
        return {"error": "expected an object with cmd"}
    cmd = message["cmd"]
    try:
        if cmd == "reset":
            obs, info = env.reset(message.get("seed", 0), message.get("config"))
            return {"observation": obs, "info": info}
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(message.get("action"))
            return {"observation": obs, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}
        if cmd == "report":
            if not harness:
                return {"error": "report is not available to agents"}
            return {"report": env.final_report()}
        return {"error": "unknown cmd"}
    except (RuntimeError, ValueError) as exc:
        return {"error": str(exc)}
    except Exception:  # сервер не должен падать от чужого ввода
        return {"error": "internal error"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="audit_env JSON-lines server")
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--harness", action="store_true", help="allow the report command")
    args = parser.parse_args(argv)
    env = AuditEnv(mode=args.mode)
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")  # чужие байты не должны ронять чтение
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = handle(env, json.loads(line), args.harness)
        except (json.JSONDecodeError, RecursionError):
            reply = {"error": "invalid JSON"}
        sys.stdout.write(json.dumps(reply, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
