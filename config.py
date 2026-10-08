"""Настройки проекта: пути и параметры из переменных окружения.

Ключи API в коде не хранятся. Они читаются из файла .env (см. .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # .env необязателен: offline-режим работает и без него
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SCENARIO_DIR = DATA_DIR / "scenarios"
MODEL_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"

MODEL_FILE = MODEL_DIR / "gateway_lgbm.txt"
META_FILE = MODEL_DIR / "gateway_meta.json"
FEATURE_REFERENCE_FILE = DATA_DIR / "feature_reference.json"

# --- LLM (любой OpenAI-совместимый API) ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))

# --- Лимиты (защита от зацикливания и зависаний) ---
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "8"))        # вызовов LLM на одного агента
AGENT_TIMEOUT_S = float(os.getenv("AGENT_TIMEOUT_S", "300"))    # секунд на одного агента
RUN_TIMEOUT_S = float(os.getenv("RUN_TIMEOUT_S", "900"))        # секунд на весь прогон
MAX_INCIDENTS = int(os.getenv("MAX_INCIDENTS", "5"))            # сколько инцидентов разбирать за прогон
MAX_TOOLS_PER_AGENT = 5                                         # требование ТЗ

# --- Шлюз ---
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.5"))    # порог вероятности атаки
