"""Streamlit UI helpers and small formatting utilities."""

from __future__ import annotations

from html import escape
import re
from typing import Any

import pandas as pd
import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
        .soc-card {
          border: 1px solid rgba(148,163,184,.22);
          background: linear-gradient(180deg, rgba(17,24,39,.96), rgba(15,23,42,.92));
          border-radius: 8px;
          padding: 1rem 1.1rem;
          box-shadow: 0 18px 45px rgba(0,0,0,.22);
          min-height: 112px;
        }
        .soc-card .label {color:#94A3B8; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em;}
        .soc-card .value {color:#F8FAFC; font-size:1.35rem; font-weight:700; margin-top:.35rem; overflow-wrap:anywhere;}
        .soc-card .sub {color:#CBD5E1; font-size:.9rem; margin-top:.35rem;}
        .badge {
          display:inline-block; padding:.22rem .55rem; border-radius:999px;
          border:1px solid rgba(148,163,184,.28); background:rgba(56,189,248,.12);
          color:#E5E7EB; font-size:.78rem; margin:.1rem .18rem .1rem 0;
        }
        .panel {
          border:1px solid rgba(148,163,184,.18); border-radius:8px; padding:1rem 1.15rem;
          background:rgba(15,23,42,.55); color:#E5E7EB;
        }
        .warning-panel {
          border:1px solid rgba(251,191,36,.35); border-radius:8px; padding:1rem 1.15rem;
          background:rgba(120,53,15,.26); color:#FEF3C7;
        }
        div[data-testid="stMetric"] {
          border: 1px solid rgba(148,163,184,.18);
          background: rgba(17,24,39,.74);
          padding: .8rem 1rem; border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, subtext: str = "") -> None:
    st.markdown(
        f"""
        <div class="soc-card">
          <div class="label">{escape(label)}</div>
          <div class="value">{escape(value)}</div>
          <div class="sub">{escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel(markdown: str, warning: bool = False) -> None:
    klass = "warning-panel" if warning else "panel"
    html = escape(markdown)
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
    html = html.replace("\n", "<br>")
    st.markdown(f'<div class="{klass}">{html}</div>', unsafe_allow_html=True)


def badges(labels: list[str]) -> None:
    html = "".join(f'<span class="badge">{escape(label)}</span>' for label in labels)
    st.markdown(html, unsafe_allow_html=True)


def distribution_frame(distribution: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"state": state, "probability": probability, "percent": f"{probability:.1%}"} for state, probability in distribution.items()]
    ).sort_values("probability", ascending=False)


def risk_level(state: str, confidence: float) -> str:
    severe_states = {"Exfiltration", "S5_Data_Exfiltration", "Privilege Escalation", "Lateral Movement", "S3_Lateral_Movement"}
    if state in severe_states and confidence >= 0.45:
        return "Critical"
    if confidence >= 0.60:
        return "High"
    if confidence >= 0.35:
        return "Medium"
    return "Low"


def mini_report_markdown() -> str:
    return """
## ATT&CK-HMM: Explainable Probabilistic Prediction of Attacker Progression

### Problem
SOC analysts often see alert streams as isolated labels, while multi-stage intrusions unfold as hidden tactical phases.

### Proposed solution
ATT&CK-HMM models alert sequences with Hidden Markov Models and explains current and next likely attack stages.

### Model profiles
The project includes an ATT&CK-Enriched eight-state profile and a Published APT-HMM six-state profile adapted to the same alert vocabulary.

### HMM inference
Forward inference estimates the current posterior distribution; Viterbi reconstructs the most likely hidden state path; next-state prediction projects the posterior through the transition matrix.

### MITRE enrichment
Predicted states and important alerts are mapped to ATT&CK techniques, producing analyst-facing rankings and Navigator-compatible layers.

### Robustness evaluation
The lab evaluates phase accuracy, next-step accuracy, top-3 next-step accuracy, confidence, and sequence likelihood under inserted noise and missing alerts.

### Limitations
This is a defensive research simulator using synthetic IDS labels. Emissions are adapted manually and MITRE mappings are approximate.

### Future work
Train emissions from real alert datasets, parse Suricata or Wazuh events, mine ATT&CK Attack Flow sequences, add labeled-campaign learning, and include analyst feedback.
"""
