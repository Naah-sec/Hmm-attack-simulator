from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.analysis_pipeline import analyze_alerts
from src.export import export_attack_navigator_layer, export_prediction_report_json, export_prediction_report_markdown
from src.model_profiles import ENRICHED_PROFILE, list_available_profiles
from src.simulator import custom_sequence_from_user_input, get_scenario_names, simulate_scenario
from src.utils import badges, distribution_frame, inject_css, metric_card, panel, risk_level
from src.visualization import alert_timeline, attack_path_graph, confidence_gauge, mitre_ranking_chart, next_state_bar_chart, probability_evolution_chart


st.set_page_config(page_title="Live Attack Simulator", layout="wide")
inject_css()
st.title("Live Attack Simulator")

with st.sidebar:
    profile = st.selectbox("Model profile", list_available_profiles(), index=0)
    scenario_name = st.selectbox("Scenario", get_scenario_names())
    noise_rate = st.slider("Noise rate", 0.0, 0.5, 0.0, 0.05)
    missing_rate = st.slider("Missing alert rate", 0.0, 0.5, 0.0, 0.05)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42)
    custom = st.checkbox("Use custom alert sequence")
    raw = st.text_area("Custom alerts", placeholder="PORT_SCAN, EXPLOIT_PUBLIC_APP, C2_BEACONING", disabled=not custom)
    run = st.button("Run analysis", type="primary")

if custom:
    alerts, unknown_tokens = custom_sequence_from_user_input(raw)
    true_states = None
    scenario_label = "Custom Sequence"
else:
    simulation = simulate_scenario(scenario_name, profile, noise_rate, missing_rate, int(seed))
    alerts = simulation["alerts"]
    true_states = simulation["true_states"]
    scenario_label = scenario_name
    unknown_tokens = []

if run or "last_analysis" not in st.session_state:
    result = analyze_alerts(profile, alerts, true_states)
    result["profile"] = profile
    result["scenario_name"] = scenario_label
    result["noise_rate"] = noise_rate
    result["missing_rate"] = missing_rate
    st.session_state["last_analysis"] = result
else:
    result = st.session_state["last_analysis"]

if unknown_tokens:
    st.warning(f"Unknown custom tokens were mapped to UNKNOWN_ALERT: {', '.join(unknown_tokens)}")

model = result["model"]
current_state, current_conf = max(result["current_distribution"].items(), key=lambda item: item[1])
next_state, next_prob = max(result["next_distribution"].items(), key=lambda item: item[1])
risk = risk_level(current_state, current_conf)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Current inferred phase", current_state, model.name)
with c2:
    metric_card("Current confidence", f"{current_conf:.0%}", result["explanation"]["confidence_label"])
with c3:
    metric_card("Predicted next phase", next_state, f"{next_prob:.0%} next-step probability")
with c4:
    metric_card("Risk level", risk, "Derived from phase and confidence")

tabs = st.tabs(["Overview", "Timeline", "Probabilities", "MITRE Mapping", "Explanation", "Raw Data"])

with tabs[0]:
    st.subheader("Observed Alert Sequence")
    badges(result["alerts"])
    st.subheader("Most Likely Hidden Path")
    badges(result["viterbi"]["path"])
    panel(result["explanation"]["executive_summary"])
    left, right = st.columns(2)
    with left:
        st.dataframe(distribution_frame(result["current_distribution"]), use_container_width=True, hide_index=True)
    with right:
        st.dataframe(distribution_frame(result["next_distribution"]), use_container_width=True, hide_index=True)

with tabs[1]:
    confidences = [float(step["confidence"]) for step in result["viterbi"]["steps"]]
    st.plotly_chart(alert_timeline(result["alerts"], result["viterbi"]["path"], result["true_states"], confidences, result["alert_mapping"]), use_container_width=True)
    st.plotly_chart(attack_path_graph(model, result["viterbi"]["path"], next_state), use_container_width=True)

with tabs[2]:
    st.plotly_chart(probability_evolution_chart(result["evolution"], model.states), use_container_width=True)
    a, b = st.columns([1, 1])
    with a:
        st.plotly_chart(confidence_gauge(current_conf), use_container_width=True)
    with b:
        st.plotly_chart(next_state_bar_chart(result["next_distribution"]), use_container_width=True)

with tabs[3]:
    st.plotly_chart(mitre_ranking_chart(result["mitre_techniques"]), use_container_width=True)
    st.dataframe(pd.DataFrame(result["mitre_techniques"]), use_container_width=True, hide_index=True)
    alert_rows = [{"alert": alert, "techniques": techniques} for alert, techniques in result["alert_mapping"].items()]
    st.dataframe(pd.DataFrame(alert_rows), use_container_width=True, hide_index=True)
    if st.button("Export Navigator layer"):
        path = export_attack_navigator_layer(result["mitre_techniques"])
        st.success(f"Saved {path}")

with tabs[4]:
    panel(result["explanation"]["analyst_explanation"])
    st.markdown("#### Mathematical Explanation")
    panel(result["explanation"]["mathematical_explanation"])
    st.dataframe(pd.DataFrame(result["explanation"]["emission_details"]), use_container_width=True, hide_index=True)
    if result["explanation"]["caveats"]:
        panel("<br>".join(result["explanation"]["caveats"]), warning=True)

with tabs[5]:
    st.dataframe(result["steps_df"], use_container_width=True, hide_index=True)
    st.json(
        {
            "current_distribution": result["current_distribution"],
            "next_distribution": result["next_distribution"],
            "viterbi": result["viterbi"],
        }
    )
    if st.button("Export prediction report"):
        json_path = export_prediction_report_json(
            result["profile"],
            result["scenario_name"],
            result["noise_rate"],
            result["missing_rate"],
            result["alerts"],
            result["viterbi"]["path"],
            result["current_distribution"],
            result["next_distribution"],
            result["mitre_techniques"],
            result["explanation"],
        )
        md_path = export_prediction_report_markdown(
            result["profile"],
            result["scenario_name"],
            result["alerts"],
            result["viterbi"]["path"],
            result["explanation"],
            result["mitre_techniques"],
        )
        st.success(f"Saved {json_path} and {md_path}")
    st.download_button("Download current analysis JSON", data=json.dumps(result["viterbi"], indent=2), file_name="viterbi_result.json")

