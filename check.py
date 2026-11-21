"""Единая команда проверки: python check.py

1. тесты (pytest);
2. 20 открытых эпизодов: корректность данных и эталона, повторная генерация даёт те же байты
   (в том же процессе и в отдельном процессе с другим PYTHONHASHSEED);
3. четыре базовые политики и порядок их наград.
Без сети, платных API и GPU. Код возврата 0 = всё в порядке.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from audit_env.generator import generate  # noqa: E402
from audit_env.grader.validate import check_episode  # noqa: E402
from audit_env.plan import all_episodes  # noqa: E402
from scripts.run_baselines import run_all, summary  # noqa: E402


def step(title):
    print(f"\n=== {title} ===", flush=True)


def plan_digest() -> str:
    """SHA-256 всех открытых и закрытых файлов 20 эпизодов плана."""
    digest = hashlib.sha256()
    for _, _, item in all_episodes():
        ep = generate(item["seed"], item["config"])
        for files in (ep.open_files(), ep.closed_files()):
            for name in sorted(files):
                digest.update(name.encode() + b"\0" + files[name].encode("utf-8"))
    return digest.hexdigest()


def judge_protocol(native_reports) -> list:
    """Протокол жюри: эпизод в папку через generate_episode, 4 политики через make_env (closed, episode_dir)
    и score_episode — награды должны совпасть с прямым прогоном; затем парная копия (перестановка записей
    и/или смена ID): данные корректны, oracle с новыми ID получает те же метрики."""
    import tempfile

    import judge_adapter as ja
    from audit_env.files import load_episode, load_truth
    from audit_env.transform import map_findings, transform_episode

    native = {(r["episode"], r["policy"]): r["reward"] for r in native_reports}
    kinds = [(True, False), (False, True), (True, True)]
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        for split, number, item in all_episodes():
            label = f"{split}/{number:02d}_{item['name']}"
            src, dst = os.path.join(tmp, f"{split}{number}"), os.path.join(tmp, f"{split}{number}_pair")
            ja.generate_episode(item["seed"], item["config"], src)
            for name in ("random_explorer", "simple_reconciler", "flag_all", "oracle"):
                got = ja.run_policy(name, src)["return"]
                if abs(got - native[(label, name)]) > 1e-9:
                    problems.append(f"{label} {name}: через адаптер {got}, напрямую {native[(label, name)]}")
            permute, remap = kinds[number % 3]
            maps = transform_episode(src, dst, permute=permute, remap=remap, seed=item["seed"])
            if check_episode(load_episode(dst, with_closed=True)):
                problems.append(f"{label}: парная копия не прошла проверки данных")
            truth = load_truth(src)
            answer = {"episode_id": truth["episode_id"], "findings": [
                {"type": i["type"], "affected_ids": i["affected_ids"], "claimed_amount_minor": i["expected_amount_minor"],
                 "currency": i["currency"], "evidence_ids": i["evidence_sets"][0], "explanation": "oracle"}
                for i in truth["issues"]]}
            if ja.score_episode(src, answer, 5) != ja.score_episode(dst, map_findings(answer, maps), 5):
                problems.append(f"{label}: oracle на парной копии получил другие метрики")
    return problems


def main() -> int:
    if "--digest" in sys.argv:  # служебный режим для проверки в отдельном процессе
        print(plan_digest())
        return 0
    failed = []

    step("1/4 Тесты")
    if subprocess.call([sys.executable, "-m", "pytest"], cwd=ROOT) != 0:
        failed.append("тесты")

    step("2/4 Эпизоды: данные, эталон, воспроизводимость")
    bad = 0
    for split, number, item in all_episodes():
        ep = generate(item["seed"], item["config"])
        problems = check_episode(ep)
        again = generate(item["seed"], item["config"])
        if ep.open_files() != again.open_files() or ep.closed_files() != again.closed_files():
            problems.append("повторная генерация дала другие данные")
        if problems:
            bad += 1
            print(f"  {split}/{number:02d} {item['name']}: {problems[:3]}")
    print(f"  проверено эпизодов: {len(all_episodes())}, с проблемами: {bad}")
    if bad:
        failed.append("эпизоды")
    # воспроизводимость между процессами: другой PYTHONHASHSEED не должен менять ни байта
    here = plan_digest()
    env = dict(os.environ, PYTHONHASHSEED="12345")
    other = subprocess.run([sys.executable, str(ROOT / "check.py"), "--digest"], cwd=ROOT, env=env,
                           capture_output=True, text=True).stdout.strip()
    same = here == other
    print("  те же байты в отдельном процессе с другим PYTHONHASHSEED:", "да" if same else "НЕТ")
    if not same:
        failed.append("воспроизводимость между процессами")

    step("3/4 Базовые политики")
    reports = run_all()
    print(summary(reports))
    mean = lambda name: sum(r["reward"] for r in reports if r["policy"] == name) / sum(1 for r in reports if r["policy"] == name)
    ok = mean("oracle") == 1.0 and mean("oracle") > mean("simple_reconciler") > max(mean("random_explorer"), mean("flag_all"))
    print("  порядок наград oracle > simple_reconciler > random/flag_all:", "да" if ok else "НЕТ")
    if not ok:
        failed.append("порядок наград")

    step("4/4 Адаптер жюри и парные проверки (judge_adapter.py)")
    problems = judge_protocol(reports)
    for line in problems[:10]:
        print("  " + line)
    print("  20 эпизодов через generate_episode и episode_dir; 4 политики в закрытом режиме; "
          "парные варианты (перестановка, смена ID, оба):", "без расхождений" if not problems else f"ПРОБЛЕМ: {len(problems)}")
    if problems:
        failed.append("адаптер жюри")

    print()
    if failed:
        print("ПРОВЕРКА НЕ ПРОЙДЕНА:", ", ".join(failed))
        return 1
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
