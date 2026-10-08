"""Структурированные сообщения, которыми обмениваются агенты (Pydantic-модели = JSON).

Правило проекта: числа в сообщениях берутся из инструментов и общего состояния,
а LLM принимает решения и пишет пояснения. Так модель не может «придумать» цифру.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Priority = Literal["critical", "high", "medium", "low"]
Confidence = Literal["high", "medium", "low"]


# ---------- Триаж ----------

class AlertGroup(BaseModel):
    """Группа алертов, посчитанная инструментом group_alerts."""
    group_id: str
    grouped_by: str
    key: str
    target: str | None = None
    alerts: int
    target_flows_total: int | None = Field(None, description="Всего потоков к цели за окно (алерты и не алерты)")
    sources: list[str]
    sources_count: int
    dst_ports_unique: int
    top_dst_ports: list[int]
    protocols: dict[str, int]
    services: dict[str, int]
    conn_states: dict[str, int]
    mean_attack_prob: float
    first_seen: str
    last_seen: str
    alerts_per_minute: float
    sample_flow_ids: list[str]


class TriageIncidentDecision(BaseModel):
    group_id: str
    priority: Priority
    hypothesis: str = Field(description="Короткая гипотеза о характере активности")
    rationale: str = Field(description="Почему это инцидент и почему такой приоритет")


class TriageNoiseDecision(BaseModel):
    group_id: str
    reason: str


class TriageDecision(BaseModel):
    """Финальный ответ LLM агента триажа (решения без чисел)."""
    incidents: list[TriageIncidentDecision] = []
    noise: list[TriageNoiseDecision] = []
    summary: str


class Incident(BaseModel):
    """Инцидент: решение агента триажа + статистика группы из инструмента."""
    incident_id: str
    priority: Priority
    hypothesis: str
    rationale: str
    group: AlertGroup


class NoiseGroup(BaseModel):
    group_id: str
    alerts: int
    reason: str


class TriageReport(BaseModel):
    """Сообщение Триаж -> следующие агенты."""
    msg_type: Literal["triage_report"] = "triage_report"
    flows_total: int
    alerts_total: int
    incidents: list[Incident]
    noise: list[NoiseGroup]
    summary: str


# ---------- Объяснение (XAI) ----------

class ExplainTask(BaseModel):
    """Сообщение Триаж -> Объяснение: один инцидент на разбор."""
    msg_type: Literal["explain_task"] = "explain_task"
    incident: Incident


class FeatureEvidence(BaseModel):
    """Вклад одного признака по SHAP (числа из инструмента shap_incident)."""
    feature: str
    mean_shap: float = Field(description="Средний вклад в логарифм шансов атаки; > 0 толкает к «атаке»")
    mean_abs_shap: float
    incident_value: str = Field(description="Типичное значение признака в инциденте")
    normal_value: str = Field(description="Типичное значение у нормального трафика")


class ExplainDecision(BaseModel):
    """Финальный ответ LLM агента объяснения (решения без чисел)."""
    key_features: list[str] = Field(min_length=3, description="Не меньше 3 имён признаков ровно как в поле feature "
                                                              "результата shap_incident, без значений и пояснений")
    behavior_pattern: str = Field(description="Что за поведение сети стоит за этими признаками")
    narrative: str = Field(description="Понятное объяснение для оператора")
    confidence: Confidence
    possible_false_positive: bool


class Explanation(BaseModel):
    """Сообщение Объяснение -> следующие агенты."""
    msg_type: Literal["explanation"] = "explanation"
    incident_id: str
    flows_analyzed: int
    base_value: float = Field(description="Базовое значение SHAP (логарифм шансов)")
    mean_model_output: float = Field(description="Средний выход модели по инциденту (логарифм шансов)")
    evidence: list[FeatureEvidence]
    behavior_pattern: str
    narrative: str
    confidence: Confidence
    possible_false_positive: bool


# ---------- Итог прогона ----------

class IncidentDossier(BaseModel):
    """Всё, что известно об инциденте к концу прогона."""
    incident: Incident
    explanation: Explanation | None = None
    status: Literal["explained", "failed", "skipped"] = "skipped"
    error: str | None = None
