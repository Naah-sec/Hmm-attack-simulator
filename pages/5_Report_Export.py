from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.export import export_attack_navigator_layer, export_prediction_report_json, export_prediction_report_markdown
from src.utils import inject_css, mini_report_markdown, panel


st.set_page_config(page_title="Report Export", layout="wide")
inject_css()
st.title("Report Export")

result = st.session_state.get("last_analysis")
if result:
    st.subheader("Last Prediction Summary")
    panel(result["explanation"]["executive_summary"])
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
    nav_path = export_attack_navigator_layer(result["mitre_techniques"])
    for path in [json_path, md_path, nav_path, Path("outputs/experiment_results.csv")]:
        if path.exists():
            st.download_button(
                f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/octet-stream",
            )
else:
    st.info("Run the Live Attack Simulator first to populate a prediction report.")

st.subheader("Generated Mini Report")
report = mini_report_markdown()
st.markdown(report)
st.download_button("Download mini_report.md", data=report, file_name="mini_report.md", mime="text/markdown")

