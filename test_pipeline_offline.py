"""Сквозной прогон конвейера в offline-режиме на всех сценариях."""
import json

import pytest

from soc import config
from soc.logging_utils import load_events
from soc.pipeline import check_expected, run_pipeline, scenario_index
from soc.state import RunState

INDEX = scenario_index()


@pytest.mark.parametrize("name", list(INDEX))
def test_scenario(name, tmp_path):
    state, run_dir = run_pipeline(config.SCENARIO_DIR / INDEX[name]["file"], scenario=name,
                                  llm_mode="offline", runs_dir=tmp_path)
    assert state.status == "completed"
    assert check_expected(state, INDEX[name]["expected_incidents"])["passed"]
    assert all(d.status == "explained" and len(d.explanation.evidence) >= 3 for d in state.dossiers)

    # состояние и отчёт сохранены, журнал полон
    assert RunState.load(run_dir).run_id == state.run_id
    assert (run_dir / "report.md").read_text(encoding="utf-8").startswith("# Отчёт")
    events = load_events(run_dir / "calls.jsonl")
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert sum(r["total"] for r in state.call_stats["agents"]) == kinds.count("llm_call") + kinds.count("tool_call")
    # агенты действительно обменялись сообщением
    handoffs = [e for e in events if e["kind"] == "message" and e["agent"] == "triage" and e["to"] == "explainer"]
    assert len(handoffs) == len(state.dossiers)


def test_numbers_come_from_tools_not_from_llm(tmp_path):
    state, run_dir = run_pipeline(config.SCENARIO_DIR / "ddos.csv", scenario="ddos",
                                  llm_mode="offline", runs_dir=tmp_path)
    shap_calls = [e for e in load_events(run_dir / "calls.jsonl")
                  if e["kind"] == "tool_call" and e["tool"] == "shap_incident"]
    from_tool = {f["feature"]: f["mean_shap"] for f in json.loads(shap_calls[0]["result"])["features"]}
    for evidence in state.dossiers[0].explanation.evidence:
        assert from_tool[evidence.feature] == evidence.mean_shap


def test_missing_file_gives_clear_error(tmp_path):
    state, _ = run_pipeline(tmp_path / "нет_такого.csv", llm_mode="offline", runs_dir=tmp_path)
    assert state.status == "failed" and "не найден" in state.errors[0]


def test_bad_file_gives_clear_error(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    state, _ = run_pipeline(bad, llm_mode="offline", runs_dir=tmp_path)
    assert state.status == "failed" and "нет столбцов" in state.errors[0]
