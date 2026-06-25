from __future__ import annotations

import streamlit as st

from src.model_profiles import list_available_profiles
from src.utils import inject_css, metric_card, panel


st.set_page_config(
    page_title="ATT&CK-HMM Simulator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.sidebar.title("ATT&CK-HMM")
st.sidebar.selectbox("Default model profile", list_available_profiles(), key="global_profile")
st.sidebar.caption("Defensive research simulator. No exploitation, scanning, or live intrusion activity is implemented.")

st.title("ATT&CK-HMM")
st.subheader("An Explainable Hidden Markov Simulator for Predicting Multi-Stage Attacker Progression from IDS Alerts")

panel(
    "This project simulates IDS alert labels and uses manually implemented Hidden Markov Model inference to estimate "
    "attacker progression, next likely phase, and related MITRE ATT&CK techniques. It is a defensive research prototype "
    "with synthetic observations only."
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Inference", "Forward + Viterbi", "Manual NumPy implementation")
with c2:
    metric_card("Profiles", "2 HMMs", "Published APT and ATT&CK-Enriched")
with c3:
    metric_card("Outputs", "Reports + Navigator", "JSON, Markdown, CSV, ATT&CK layer")
with c4:
    metric_card("Evaluation", "Robustness Lab", "Noise and missing-alert experiments")

st.markdown("### Research Architecture")
st.markdown(
    """
```text
Synthetic IDS alert stream
        |
        v
Alert catalog + scenario generator ---- noise / missing alert controls
        |
        v
HMM profile selector
  - Published APT-HMM 6-state transition matrix
  - ATT&CK-Enriched 8-state model
        |
        v
Manual HMM inference
  - Forward posterior distribution
  - Viterbi most likely hidden path
  - Next-state prediction
        |
        v
SOC dashboard
  - Timeline, probability evolution, attack graph
  - MITRE technique ranking and Navigator layer
  - Analyst explanation and exportable reports
```
"""
)

st.markdown("### Quick Launch")
st.info("Open **Live Attack Simulator** from the left navigation, select a scenario, add noise if desired, and run analysis.")

