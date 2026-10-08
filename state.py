"""Явное состояние прогона. Сохраняется в runs/<run_id>/state.json после каждого шага."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from soc.messages import IncidentDossier, TriageReport


class RunState(BaseModel):
    run_id: str
    traffic_source: str
    scenario: str | None = None
    llm_model: str
    status: Literal["running", "completed", "completed_with_errors", "failed"] = "running"
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    limits: dict = {}
    gateway: dict = {}                       # что шлюз передал агентам: число потоков и алертов
    triage: TriageReport | None = None
    dossiers: list[IncidentDossier] = []
    errors: list[str] = []
    call_stats: dict = {}

    def save(self, run_dir: Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "state.json").write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, run_dir: Path) -> "RunState":
        return cls.model_validate_json((Path(run_dir) / "state.json").read_text(encoding="utf-8"))
