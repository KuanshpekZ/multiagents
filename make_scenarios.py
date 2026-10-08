"""Сборка тестовых сценариев из отложенной части TON_IoT (модель шлюза её не видела).

Каждый сценарий — CSV с нормальным трафиком и, возможно, атакой на конкретное устройство.
В датасете нет времени, поэтому время задаём сами: норма идёт равномерно 10 минут,
атака — плотной серией внутри этого окна (столбец second).

Запуск:  python scripts/make_scenarios.py   (после scripts/train_gateway.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from soc import config  # noqa: E402

HOLDOUT_FILE = config.RAW_DIR / "holdout.csv"
WINDOW_S = 600
SEED = 7

# attack: тип из TON_IoT, цель, число потоков, начало и длительность серии (сек)
SCENARIOS = {
    "ddos": {
        "title": "DDoS на устройство 192.168.1.184",
        "normal": 400,
        "attacks": [{"type": "ddos", "dst_ip": "192.168.1.184", "n": 300, "start": 180, "length": 120}],
    },
    "scanning": {
        "title": "Сканирование портов устройства 192.168.1.49",
        "normal": 400,
        "attacks": [{"type": "scanning", "dst_ip": "192.168.1.49", "n": 250, "start": 240, "length": 90}],
    },
    "password": {
        "title": "Подбор паролей к устройству 192.168.1.190",
        "normal": 400,
        "attacks": [{"type": "password", "dst_ip": "192.168.1.190", "n": 250, "start": 120, "length": 240}],
    },
    "mixed": {
        "title": "Две атаки одновременно: DDoS на .184 и сканирование .49",
        "normal": 400,
        "attacks": [
            {"type": "ddos", "dst_ip": "192.168.1.184", "n": 300, "start": 120, "length": 120},
            {"type": "scanning", "dst_ip": "192.168.1.49", "n": 120, "start": 360, "length": 90},
        ],
    },
    "normal": {
        "title": "Только нормальный трафик (инцидентов быть не должно)",
        "normal": 500,
        "attacks": [],
    },
}


def main() -> None:
    if not HOLDOUT_FILE.exists():
        sys.exit(f"Нет файла {HOLDOUT_FILE}. Сначала запустите scripts/train_gateway.py")
    holdout = pd.read_csv(HOLDOUT_FILE, low_memory=False)
    rng = np.random.default_rng(SEED)
    config.SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    index = {}

    for number, (name, spec) in enumerate(SCENARIOS.items()):
        # у каждого сценария своя выборка нормального трафика
        normal = holdout[holdout["type"] == "normal"].sample(spec["normal"], random_state=SEED + number).copy()
        normal["second"] = rng.integers(0, WINDOW_S, len(normal))
        parts = [normal]
        for attack in spec["attacks"]:
            pool = holdout[(holdout["type"] == attack["type"]) & (holdout["dst_ip"] == attack["dst_ip"])]
            chosen = pool.sample(attack["n"], random_state=SEED).copy()
            chosen["second"] = rng.integers(attack["start"], attack["start"] + attack["length"], len(chosen))
            parts.append(chosen)
        traffic = pd.concat(parts).sort_values("second", kind="stable").reset_index(drop=True)
        traffic.to_csv(config.SCENARIO_DIR / f"{name}.csv", index=False)
        index[name] = {
            "title": spec["title"],
            "file": f"{name}.csv",
            "flows": int(len(traffic)),
            # Ожидаемый результат. Используется только для проверки, агентам не передаётся.
            "expected_incidents": [{"target": a["dst_ip"], "attack_type": a["type"]} for a in spec["attacks"]],
        }
        print(f"{name:10s} {len(traffic):4d} потоков  {spec['title']}")

    (config.SCENARIO_DIR / "scenarios.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
