"""Offline-режим: замена LLM фиксированными правилами. Сеть и ключ API не нужны.

Зачем он нужен:
  - проверить, что система запускается и агенты обмениваются сообщениями, без расхода запросов;
  - автотесты;
  - запасной вариант на случай, если API недоступен во время показа.

Важно: это НЕ языковая модель. Правила ниже жёстко заданы, поэтому гипотезы и объяснения
шаблонные. Настоящая работа агентов показывается в режиме api.
Интерфейс тот же, что у OpenAICompatLLM: тот же цикл агента, те же инструменты и проверки.
"""
from __future__ import annotations

import json

from soc.llm import LLMResponse, ToolCall


def _history(messages: list[dict]) -> list[dict]:
    """Какие инструменты уже вызывались и что они вернули."""
    calls, order = {}, []
    for m in messages:
        if m["role"] == "assistant":
            for c in m.get("tool_calls", []):
                calls[c["id"]] = {"tool": c["function"]["name"],
                                  "args": json.loads(c["function"]["arguments"]), "result": None}
                order.append(c["id"])
        elif m["role"] == "tool" and m["tool_call_id"] in calls:
            calls[m["tool_call_id"]]["result"] = json.loads(m["content"])
    return [calls[i] for i in order]


def _result(history: list[dict], tool: str) -> dict | None:
    for record in reversed(history):
        if record["tool"] == tool and record["result"] is not None and "error" not in record["result"]:
            return record["result"]
    return None


def _task(messages: list[dict]) -> dict:
    text = next(m["content"] for m in messages if m["role"] == "user")
    return json.loads(text[text.find("{"):])


class OfflineLLM:
    model = "offline-rules"

    def __init__(self):
        self._n = 0

    def chat(self, agent: str, messages: list[dict], tools: list[dict] | None) -> LLMResponse:
        policy = {"triage": self._triage, "explainer": self._explainer}.get(agent)
        if policy is None:
            raise ValueError(f"В offline-режиме нет правил для агента «{agent}»")
        step = policy(messages, _history(messages))
        if isinstance(step, list) and tools:
            calls = []
            for name, args in step:
                self._n += 1
                calls.append(ToolCall(id=f"offline_{self._n}", name=name, arguments=args))
            return LLMResponse(content=None, tool_calls=calls)
        if isinstance(step, list):   # инструменты уже недоступны (последний шаг)
            step = policy(messages, _history(messages), force_final=True)
        return LLMResponse(content=json.dumps(step, ensure_ascii=False))

    # ---------------- Триаж ----------------

    def _triage(self, messages, history, force_final: bool = False):
        summary = _result(history, "fetch_gateway_alerts")
        if summary is None and not force_final:
            return [("fetch_gateway_alerts", {})]
        if not summary or summary["alerts_total"] == 0:
            return {"incidents": [], "noise": [], "summary": "Алертов за окно нет, инцидентов не выявлено."}

        grouped = _result(history, "group_alerts")
        if grouped is None:
            if force_final:
                return {"incidents": [], "noise": [], "summary": "Не удалось сгруппировать алерты."}
            return [("group_alerts", {"by": "dst_ip"})]

        groups = grouped["groups"]
        big = [g for g in groups if g["alerts"] >= 10]
        if big and _result(history, "traffic_baseline") is None and not force_final:
            return [("traffic_baseline", {"dst_ip": big[0]["target"]}),
                    ("get_flow_features", {"group_id": big[0]["group_id"], "n": 3})]

        incidents, noise = [], []
        for g in groups:
            if g["alerts"] < 10:
                noise.append({"group_id": g["group_id"],
                              "reason": f"Единичные алерты ({g['alerts']}) без общей картины; у шлюза около "
                                        "1% ложных срабатываний."})
                continue
            if g["alerts"] >= 200 and g["sources_count"] >= 2:
                priority = "critical"
            elif g["alerts"] >= 100:
                priority = "high"
            else:
                priority = "medium"
            top_state = next(iter(g["conn_states"]), "")
            if g["dst_ports_unique"] >= 20:
                hypothesis = "Перебор множества портов одного устройства, похоже на сканирование."
            elif top_state == "SF":
                hypothesis = ("Множество завершённых однотипных обращений к одному сервису, похоже на "
                              "автоматический перебор запросов.")
            elif g["alerts_per_minute"] >= 60 and g["dst_ports_unique"] <= 3:
                hypothesis = ("Массовые однотипные соединения на один-два порта, похоже на попытку "
                              "перегрузить сервис.")
            else:
                hypothesis = "Серия подозрительных соединений с одним устройством."
            incidents.append({
                "group_id": g["group_id"], "priority": priority, "hypothesis": hypothesis,
                "rationale": (f"{g['alerts']} алертов из {g['target_flows_total']} потоков к {g['target']}, "
                              f"источников: {g['sources_count']}, уникальных портов: {g['dst_ports_unique']}, "
                              f"{g['alerts_per_minute']} алертов в минуту, средняя вероятность атаки "
                              f"{g['mean_attack_prob']}.")})
        return {"incidents": incidents, "noise": noise,
                "summary": f"Из {summary['alerts_total']} алертов выделено инцидентов: {len(incidents)}, "
                           f"групп отнесено к шуму: {len(noise)}."}

    # ---------------- Объяснение ----------------

    def _explainer(self, messages, history, force_final: bool = False):
        incident = _task(messages)["incident"]
        shap_result = _result(history, "shap_incident")
        if shap_result is None:
            if force_final:
                return {"key_features": [], "behavior_pattern": "", "narrative": "", "confidence": "low",
                        "possible_false_positive": False}
            return [("shap_incident", {"incident_id": incident["incident_id"], "top_k": 5})]

        rows = shap_result["features"]
        strong = [r for r in rows if r["mean_abs_shap"] >= 0.1][:5]
        key = strong if len(strong) >= 3 else rows[:3]
        names = [r["feature"] for r in key]

        reference = _result(history, "feature_reference")
        if reference is None and not force_final:
            return [("feature_reference", {"features": names[:3]}),
                    ("shap_flow", {"flow_id": incident["group"]["sample_flow_ids"][0]})]

        example = _result(history, "shap_flow")
        example_agrees = bool(example) and example["contributions"][0]["feature"] in names
        parts = [f"{r['feature']} (в инциденте: {r['incident_value']}; в норме: {r['normal_value']}; "
                 f"средний вклад {r['mean_shap']:+.2f})" for r in key]
        meanings = (reference or {}).get("features", {})
        pattern = " ".join(f"{name}: {meanings[name]['threat_indicator']}" for name in names[:2] if name in meanings)
        output = shap_result["mean_model_output"]
        return {
            "key_features": names,
            "behavior_pattern": pattern or "Значения главных признаков заметно отличаются от нормального трафика.",
            "narrative": ("Модель шлюза отнесла потоки к атаке в основном из-за признаков: "
                          + "; ".join(parts) + f". Проанализировано потоков: {shap_result['flows_analyzed']}."),
            "confidence": "high" if output >= 2.0 and shap_result["flows_analyzed"] >= 20 and example_agrees
                          else "medium",
            "possible_false_positive": output < 0.5,
        }
