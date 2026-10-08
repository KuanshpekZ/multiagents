"""Командная строка проекта.

  python run.py scenarios                       список готовых сценариев
  python run.py run --scenario ddos             прогон сценария (LLM из .env, а без ключа — offline)
  python run.py run --scenario ddos --llm offline
  python run.py run --file путь/к/трафику.csv   свой файл трафика
  python run.py stats runs/<папка>              нагрузка по агентам из журнала
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from soc import config  # noqa: E402
from soc.logging_utils import format_stats, load_stats  # noqa: E402
from soc.pipeline import check_expected, run_pipeline, scenario_index  # noqa: E402
from soc.report import PRIORITY_RU  # noqa: E402


def cmd_scenarios(_args) -> int:
    index = scenario_index()
    if not index:
        print("Сценариев нет. Запустите: python scripts/make_scenarios.py")
        return 1
    for name, info in index.items():
        print(f"{name:10s} {info['flows']:5d} потоков  {info['title']}")
    return 0


def cmd_run(args) -> int:
    index = scenario_index()
    if args.scenario:
        if args.scenario not in index:
            print(f"Нет сценария «{args.scenario}». Доступны: {', '.join(index) or 'нет'}")
            return 1
        traffic = config.SCENARIO_DIR / index[args.scenario]["file"]
    else:
        traffic = Path(args.file)

    try:
        state, run_dir = run_pipeline(traffic, scenario=args.scenario, llm_mode=args.llm,
                                      max_incidents=args.max_incidents)
    except Exception as e:  # noqa: BLE001 — последний рубеж: понятное сообщение вместо трассировки
        print(f"Прогон не удался: {e}")
        return 1

    print(f"\nПрогон {state.run_id}: {state.status}  (LLM: {state.llm_model})")
    if state.llm_model == "offline-rules":
        print("Внимание: offline-режим, решения по фиксированным правилам, без языковой модели.")
    if state.gateway:
        print(f"Шлюз: потоков {state.gateway['flows_total']}, алертов {state.gateway['alerts_total']}")
    if state.triage:
        print(f"Триаж: {state.triage.summary}")
    for d in state.dossiers:
        g = d.incident.group
        print(f"\n[{d.incident.incident_id}] приоритет: {PRIORITY_RU[d.incident.priority]}, цель {g.target}, "
              f"алертов {g.alerts}")
        print(f"  Гипотеза: {d.incident.hypothesis}")
        if d.explanation:
            top = ", ".join(f"{e.feature} ({e.mean_shap:+.2f})" for e in d.explanation.evidence)
            print(f"  Главные признаки (SHAP): {top}")
            print(f"  Объяснение: {d.explanation.narrative}")
        elif d.error:
            print(f"  Объяснение не получено: {d.error}")
    for error in state.errors:
        print(f"Ошибка: {error}")

    if args.scenario:
        check = check_expected(state, index[args.scenario]["expected_incidents"])
        verdict = "совпало" if check["passed"] else "НЕ совпало"
        print(f"\nПроверка сценария: {verdict}. Ожидались цели {check['expected_targets']}, "
              f"найдены {check['found_targets']}")

    print("\nНагрузка по агентам:")
    print(format_stats(state.call_stats))
    print(f"\nФайлы прогона: {run_dir}  (report.md, state.json, calls.jsonl)")
    return 0 if state.status == "completed" else 2


def cmd_stats(args) -> int:
    path = Path(args.run_dir)
    path = path / "calls.jsonl" if path.is_dir() else path
    if not path.exists():
        print(f"Журнал не найден: {path}")
        return 1
    print(format_stats(load_stats(path)))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Многоагентный разбор сетевых инцидентов на IoT-шлюзе")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scenarios", help="список сценариев").set_defaults(func=cmd_scenarios)

    run = sub.add_parser("run", help="запустить разбор")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--scenario", help="имя готового сценария")
    source.add_argument("--file", help="CSV с трафиком в формате TON_IoT")
    run.add_argument("--llm", choices=["auto", "api", "offline"], default="auto",
                     help="api — LLM из .env; offline — правила без LLM; auto — api, если задан ключ")
    run.add_argument("--max-incidents", type=int, default=config.MAX_INCIDENTS)
    run.set_defaults(func=cmd_run)

    stats = sub.add_parser("stats", help="нагрузка по агентам")
    stats.add_argument("run_dir", help="папка прогона или файл calls.jsonl")
    stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
