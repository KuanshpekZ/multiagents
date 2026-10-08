"""Конвейер прототипа: Шлюз -> Триаж -> Объяснение (по каждому инциденту) -> отчёт.

Порядок работы задаёт этот код, а не отдельный агент-оркестратор.
Агенты общаются типизированными сообщениями (soc/messages.py), всё пишется в журнал.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from soc import config
from soc.agent import AgentError
from soc.agents.explainer import ExplainerAgent
from soc.agents.triage import TriageAgent, TriageTask
from soc.gateway.simulator import Gateway
from soc.llm import LLM, make_llm
from soc.logging_utils import CallLogger, load_stats
from soc.messages import ExplainTask, IncidentDossier
from soc.report import build_report
from soc.state import RunState
from soc.tools.triage_tools import TriageToolbox
from soc.tools.xai_tools import XaiToolbox


def scenario_index() -> dict:
    path = config.SCENARIO_DIR / "scenarios.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_pipeline(traffic_file: Path, scenario: str | None = None, llm: LLM | None = None,
                 llm_mode: str = "auto", max_incidents: int = config.MAX_INCIDENTS,
                 runs_dir: Path = config.RUNS_DIR) -> tuple[RunState, Path]:
    """Полный прогон. Возвращает итоговое состояние и папку с результатами."""
    started = time.monotonic()
    traffic_file = Path(traffic_file)
    try:   # в журнале и состоянии показываем путь относительно проекта
        shown_path = str(traffic_file.resolve().relative_to(config.ROOT))
    except ValueError:
        shown_path = str(traffic_file)
    llm = llm or make_llm(llm_mode)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + (scenario or Path(traffic_file).stem)
    run_dir = Path(runs_dir) / run_id
    logger = CallLogger(run_dir / "calls.jsonl", run_id)
    state = RunState(run_id=run_id, traffic_source=shown_path, scenario=scenario, llm_model=llm.model,
                     limits={"agent_max_steps": config.AGENT_MAX_STEPS, "agent_timeout_s": config.AGENT_TIMEOUT_S,
                             "run_timeout_s": config.RUN_TIMEOUT_S, "max_incidents": max_incidents})
    logger.event("run_start", traffic_source=shown_path, llm_model=llm.model, limits=state.limits)

    def finish(status: str) -> tuple[RunState, Path]:
        state.status = status
        state.finished_at = datetime.now().isoformat(timespec="seconds")
        state.call_stats = load_stats(logger.path)
        state.save(run_dir)
        (run_dir / "report.md").write_text(build_report(state), encoding="utf-8")
        logger.event("run_end", status=status, seconds=round(time.monotonic() - started, 2))
        return state, run_dir

    # 1. Шлюз (не агент): лёгкая модель проверяет каждый поток
    try:
        gateway = Gateway()
        store = gateway.replay(Path(traffic_file))
    except (FileNotFoundError, ValueError) as e:
        state.errors.append(f"Шлюз: {e}")
        logger.event("error", stage="gateway", error=str(e))
        return finish("failed")
    state.gateway = {"flows_total": len(store.flows), "alerts_total": len(store.alerts),
                     "alert_threshold": store.threshold, "features": gateway.features}
    logger.event("gateway_done", **state.gateway)
    state.save(run_dir)

    # 2. Агент триажа
    task = TriageTask(traffic_source=Path(traffic_file).name)
    logger.message("pipeline", "triage", task.msg_type, task.model_dump())
    try:
        triage = TriageAgent(llm, logger, TriageToolbox(store, gateway.meta))
        state.triage = triage.run(task)
    except AgentError as e:
        state.errors.append(str(e))
        return finish("failed")
    state.dossiers = [IncidentDossier(incident=i) for i in state.triage.incidents]
    state.save(run_dir)

    # 3. Агент объяснения: по одному сообщению на каждый инцидент, в порядке приоритета
    xai_toolbox = XaiToolbox(store, gateway, state.triage.incidents)
    for dossier in state.dossiers[:max_incidents]:
        if time.monotonic() - started > config.RUN_TIMEOUT_S:
            dossier.error = f"превышен лимит времени прогона {config.RUN_TIMEOUT_S:.0f} с"
            state.errors.append(f"{dossier.incident.incident_id}: {dossier.error}")
            continue
        message = ExplainTask(incident=dossier.incident)
        logger.message("triage", "explainer", message.msg_type, message.model_dump())
        try:
            explainer = ExplainerAgent(llm, logger, xai_toolbox)
            dossier.explanation = explainer.run(message)
            dossier.status = "explained"
            logger.message("explainer", "pipeline", dossier.explanation.msg_type,
                           dossier.explanation.model_dump())
        except AgentError as e:   # сбой на одном инциденте не останавливает остальные
            dossier.status, dossier.error = "failed", str(e)
            state.errors.append(f"{dossier.incident.incident_id}: {e}")
        state.save(run_dir)

    return finish("completed_with_errors" if state.errors else "completed")


def check_expected(state: RunState, expected_incidents: list[dict]) -> dict:
    """Сверка с ожидаемым результатом сценария. Агенты этих данных не видят."""
    found = {d.incident.group.target for d in state.dossiers}
    expected = {e["target"] for e in expected_incidents}
    return {"expected_targets": sorted(expected), "found_targets": sorted(t for t in found if t),
            "missed": sorted(expected - found), "extra": sorted(t for t in found - expected if t),
            "passed": found == expected}
