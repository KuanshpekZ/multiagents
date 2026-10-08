"""Шлюз и инструменты: считают верно и не раскрывают агентам правильные ответы."""
import pandas as pd
import pytest

from soc import config
from soc.messages import Incident
from soc.tools.triage_tools import TriageToolbox, group_flow_ids
from soc.tools.xai_tools import XaiToolbox

TARGET = "192.168.1.184"


def test_gateway_detects_attack_and_hides_truth(ddos_store):
    truth = pd.read_csv(config.SCENARIO_DIR / "ddos.csv")["type"]
    attack = (truth != "normal").to_numpy()
    assert ddos_store.flows["is_alert"].to_numpy()[attack].mean() > 0.95
    assert "label" not in ddos_store.flows.columns and "type" not in ddos_store.flows.columns


def test_group_alerts_finds_target(ddos_store, gateway):
    toolbox = TriageToolbox(ddos_store, gateway.meta)
    result = toolbox.group_alerts("dst_ip")
    groups = result["groups"]
    assert groups[0]["target"] == TARGET and groups[0]["alerts"] >= 280
    assert sum(g["alerts"] for g in groups) == len(ddos_store.alerts)
    # у каждого кода conn_state из групп есть расшифровка из справочника
    assert set(groups[0]["conn_states"]) <= set(result["conn_state_meaning"])
    with pytest.raises(ValueError):
        toolbox.group_alerts("wrong_key")
    with pytest.raises(ValueError):
        toolbox.traffic_baseline("10.0.0.1")


def test_shap_tools(ddos_store, gateway):
    toolbox = TriageToolbox(ddos_store, gateway.meta)
    toolbox.group_alerts("dst_ip")
    group = toolbox.groups[f"dst_ip={TARGET}"]
    incident = Incident(incident_id="INC-001", priority="high", hypothesis="h", rationale="r", group=group)
    xai = XaiToolbox(ddos_store, gateway, [incident])

    result = xai.shap_incident("INC-001", top_k=5)
    assert len(result["features"]) == 5 and result["features_total"] == 11
    assert result["mean_model_output"] > result["base_value"]          # модель склоняется к «атаке»
    assert result["flows_analyzed"] == min(200, len(group_flow_ids(ddos_store, group)))

    flow = xai.shap_flow(group.sample_flow_ids[0])
    total = sum(c["shap"] for c in flow["contributions"]) + flow["base_value"]
    assert abs(total - flow["model_output"]) < 0.02                    # SHAP складывается в выход модели

    assert "unknown" in xai.feature_reference(["conn_state", "no_such_feature"])
    with pytest.raises(ValueError):
        xai.shap_incident("INC-999")
