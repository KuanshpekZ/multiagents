"""Обучение модели шлюза (LightGBM) по методике диссертации на датасете TON_IoT.

Шаги повторяют диссертацию:
  1. исключаем идентификаторы, которые легко подделать (IP-адреса и порты);
  2. отбираем 11 самых важных признаков по важности Random Forest (только на обучающей части);
  3. обучаем лёгкую LightGBM (50 деревьев глубиной до 5) с весами классов;
  4. делим данные 80/20 со стратификацией.

Запуск:  python scripts/train_gateway.py
Нужен файл data/raw/train_test_network.csv (TON_IoT, сетевая часть).
Результат: models/gateway_lgbm.txt, models/gateway_meta.json, data/raw/holdout.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from soc import config  # noqa: E402

RAW_FILE = config.RAW_DIR / "train_test_network.csv"
HOLDOUT_FILE = config.RAW_DIR / "holdout.csv"

# Идентификаторы, которые атакующий легко подделывает (как в диссертации)
ID_COLUMNS = ["src_ip", "dst_ip", "src_port", "dst_port"]
# Строки с содержимым запросов: по сути тоже идентификаторы конкретного стенда
CONTENT_COLUMNS = ["dns_query", "ssl_subject", "ssl_issuer", "http_uri", "http_user_agent"]
TARGET_COLUMNS = ["label", "type"]
TOP_N = 11
SEED = 42


def main() -> None:
    if not RAW_FILE.exists():
        sys.exit(f"Не найден файл {RAW_FILE}. Положите туда train_test_network.csv из TON_IoT.")

    df = pd.read_csv(RAW_FILE, low_memory=False)
    df.columns = df.columns.str.strip()
    candidates = [c for c in df.columns if c not in ID_COLUMNS + CONTENT_COLUMNS + TARGET_COLUMNS]

    # Категориальные признаки кодируем числами; словари сохраняем, чтобы потом расшифровывать
    category_maps: dict[str, dict[str, int]] = {}
    X = pd.DataFrame(index=df.index)
    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            X[col] = df[col]
        else:
            values = sorted(df[col].astype(str).str.strip().unique())
            category_maps[col] = {v: i for i, v in enumerate(values)}
            X[col] = df[col].astype(str).str.strip().map(category_maps[col])
    y = df["label"].astype(int)

    idx_train, idx_test = train_test_split(
        df.index, test_size=0.2, random_state=SEED, stratify=df["type"])
    X_train, X_test, y_train, y_test = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]

    # Отбор признаков: важность Random Forest, только обучающая часть
    rf = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1).fit(X_train, y_train)
    importance = pd.Series(rf.feature_importances_, index=candidates).sort_values(ascending=False)
    features = importance.index[:TOP_N].tolist()

    started = time.time()
    model = lgb.LGBMClassifier(n_estimators=50, max_depth=5, learning_rate=0.1,
                               class_weight="balanced", random_state=SEED, verbose=-1)
    model.fit(X_train[features], y_train)
    train_seconds = time.time() - started

    proba = model.predict_proba(X_test[features])[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    batch = X_test[features].to_numpy()
    started = time.time()
    model.booster_.predict(batch)
    ms_per_flow = (time.time() - started) / len(batch) * 1000

    detection_by_type = (pd.DataFrame({"type": df.loc[idx_test, "type"].values, "pred": pred})
                         .groupby("type")["pred"].mean().round(4).to_dict())

    config.MODEL_DIR.mkdir(exist_ok=True)
    model.booster_.save_model(str(config.MODEL_FILE))

    # Эталон нормы: типичные значения признаков у нормального трафика (для объяснений)
    normal = X_train.loc[y_train == 0, features]
    baseline = {}
    for col in features:
        if col in category_maps:
            code = int(normal[col].mode().iloc[0])
            decode = {v: k for k, v in category_maps[col].items()}
            baseline[col] = {"kind": "categorical", "typical": decode[code]}
        else:
            baseline[col] = {"kind": "numeric",
                             "median": float(normal[col].median()),
                             "p90": float(normal[col].quantile(0.9)),
                             "train_min": float(X_train[col].min()),
                             "train_max": float(X_train[col].max())}

    meta = {
        "dataset": "TON_IoT (network), train_test_network.csv",
        "features": features,
        "category_maps": {c: category_maps[c] for c in features if c in category_maps},
        "excluded_identifiers": ID_COLUMNS,
        "excluded_content_fields": CONTENT_COLUMNS,
        "feature_importance_rf": {k: round(float(v), 4) for k, v in importance.head(TOP_N).items()},
        "baseline_normal": baseline,
        "params": {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.1, "class_weight": "balanced"},
        "metrics_holdout": {
            "rows_train": int(len(idx_train)), "rows_test": int(len(idx_test)),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred)), 4),
            "recall": round(float(recall_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "false_positive_rate": round(float(fp / (fp + tn)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "detection_rate_by_type": detection_by_type,
            "train_seconds": round(train_seconds, 2),
            "ms_per_flow_batch": round(ms_per_flow, 5),
            "model_size_kb": round(config.MODEL_FILE.stat().st_size / 1024, 1),
        },
    }
    config.META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Отложенная часть: из неё собираются тестовые сценарии, модель её не видела
    df.loc[idx_test].to_csv(HOLDOUT_FILE, index=False)

    print("Признаки:", features)
    print(json.dumps(meta["metrics_holdout"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
