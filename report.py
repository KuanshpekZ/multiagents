"""Итоговый отчёт по прогону. Собирается шаблоном из состояния, без LLM."""
from __future__ import annotations

from soc.logging_utils import format_stats
from soc.state import RunState

PRIORITY_RU = {"critical": "критический", "high": "высокий", "medium": "средний", "low": "низкий"}
CONFIDENCE_RU = {"high": "высокая", "medium": "средняя", "low": "низкая"}


def build_report(state: RunState) -> str:
    lines = [f"# Отчёт по прогону {state.run_id}", "",
             f"- Источник трафика: `{state.traffic_source}`",
             f"- Модель LLM: `{state.llm_model}`",
             f"- Статус: {state.status}",
             f"- Шлюз проверил потоков: {state.gateway.get('flows_total', '—')}, "
             f"поднял алертов: {state.gateway.get('alerts_total', '—')}", ""]

    if state.triage is None:
        lines += ["Триаж не выполнен.", ""]
    else:
        lines += ["## Триаж", "", state.triage.summary, ""]
        if not state.triage.incidents:
            lines += ["Инцидентов не выявлено.", ""]
        for noise in state.triage.noise:
            lines.append(f"- Шум: `{noise.group_id}`, алертов: {noise.alerts}. {noise.reason}")
        if state.triage.noise:
            lines.append("")

    for dossier in state.dossiers:
        inc, g = dossier.incident, dossier.incident.group
        lines += [f"## {inc.incident_id}: приоритет {PRIORITY_RU[inc.priority]}", "",
                  f"- Цель: {g.target or '—'}; источники ({g.sources_count}): {', '.join(g.sources)}",
                  f"- Алертов: {g.alerts} из {g.target_flows_total} потоков к цели; "
                  f"{g.alerts_per_minute} в минуту; уникальных портов: {g.dst_ports_unique}",
                  f"- Окно: {g.first_seen} — {g.last_seen}",
                  f"- Гипотеза триажа: {inc.hypothesis}",
                  f"- Обоснование: {inc.rationale}", ""]
        exp = dossier.explanation
        if exp is None:
            lines += [f"Объяснение не получено: {dossier.error or 'инцидент не разбирался'}.", ""]
            continue
        lines += ["### Почему модель считает это атакой", "", exp.narrative, "",
                  f"Поведение: {exp.behavior_pattern}", "",
                  "| Признак | Средний вклад SHAP | В инциденте | В норме |", "| --- | --- | --- | --- |"]
        for e in exp.evidence:
            lines.append(f"| {e.feature} | {e.mean_shap:+.3f} | {e.incident_value} | {e.normal_value} |")
        lines += ["", f"Уверенность: {CONFIDENCE_RU[exp.confidence]}. "
                      f"Возможное ложное срабатывание: {'да' if exp.possible_false_positive else 'нет'}. "
                      f"Проанализировано потоков: {exp.flows_analyzed}.", ""]

    if state.errors:
        lines += ["## Ошибки", ""] + [f"- {e}" for e in state.errors] + [""]
    if state.call_stats:
        lines += ["## Нагрузка по агентам", "", "```", format_stats(state.call_stats), "```", ""]
    return "\n".join(lines)
