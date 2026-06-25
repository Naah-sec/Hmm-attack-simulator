from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis_pipeline import analyze_alerts
from src.config_loader import load_mitre_mapping
from src.export import export_attack_navigator_layer
from src.mitre_mapper import build_navigator_technique_entries
from src.model_profiles import list_available_profiles
from src.simulator import get_scenario_names, simulate_scenario
from src.utils import inject_css, panel
from src.visualization import mitre_ranking_chart


st.set_page_config(page_title="MITRE ATT&CK View", layout="wide")
inject_css()
st.title("MITRE ATT&CK View")

mapping = load_mitre_mapping()
profile = st.sidebar.selectbox("Profile", list_available_profiles())
search = st.sidebar.text_input("Search technique ID or name")

rows = []
for state, technique_ids in mapping["profiles"][profile].items():
    for tid in technique_ids:
        rows.append(
            {
                "state": state,
                "technique_id": tid,
                "technique_name": mapping["technique_names"].get(tid, "Unknown technique"),
                "why": f"Mapped because {state} represents this phase in {profile}.",
            }
        )
state_df = pd.DataFrame(rows)
alert_rows = []
for alert, technique_ids in mapping["alert_to_techniques"].items():
    alert_rows.append(
        {
            "alert": alert,
            "techniques": ", ".join(technique_ids) or "None",
            "technique_names": ", ".join(mapping["technique_names"].get(tid, tid) for tid in technique_ids) or "None",
        }
    )
alert_df = pd.DataFrame(alert_rows)

if search:
    needle = search.lower()
    state_df = state_df[state_df.apply(lambda row: needle in " ".join(map(str, row)).lower(), axis=1)]
    alert_df = alert_df[alert_df.apply(lambda row: needle in " ".join(map(str, row)).lower(), axis=1)]

st.subheader("State-to-Technique Mapping")
st.dataframe(state_df, hide_index=True, use_container_width=True)

st.subheader("Alert-to-Technique Mapping")
st.dataframe(alert_df, hide_index=True, use_container_width=True)

st.subheader("Technique Score Preview")
scenario = st.selectbox("Sample scenario", get_scenario_names())
sample = simulate_scenario(scenario, profile)
analysis = analyze_alerts(profile, sample["alerts"], sample["true_states"])
st.plotly_chart(mitre_ranking_chart(analysis["mitre_techniques"]), use_container_width=True)
st.dataframe(pd.DataFrame(analysis["mitre_techniques"]), hide_index=True, use_container_width=True)

with st.expander("Navigator Export Preview", expanded=False):
    st.json({"techniques": build_navigator_technique_entries(analysis["mitre_techniques"][:8])})
    if st.button("Save Navigator preview"):
        path = export_attack_navigator_layer(analysis["mitre_techniques"])
        panel(f"Saved preview layer to `{path}`.")
