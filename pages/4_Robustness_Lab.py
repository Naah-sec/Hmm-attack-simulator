from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluator import noise_robustness_experiment, profile_robustness_comparison
from src.model_profiles import list_available_profiles
from src.simulator import get_scenario_names
from src.utils import inject_css, panel
from src.visualization import robustness_accuracy_chart, robustness_heatmap


st.set_page_config(page_title="Robustness Lab", layout="wide")
inject_css()
st.title("Robustness Lab")

with st.sidebar:
    mode = st.radio("Experiment mode", ["Single profile", "Compare both HMM profiles"])
    profile = st.selectbox("Profile", list_available_profiles(), disabled=mode != "Single profile")
    scenarios = st.multiselect("Scenarios", get_scenario_names(), default=get_scenario_names()[:3])
    noise_rates = st.multiselect("Noise rates", [0.0, 0.1, 0.2, 0.3, 0.4], default=[0.0, 0.2])
    missing_rates = st.multiselect("Missing-alert rates", [0.0, 0.1, 0.2, 0.3], default=[0.0])
    seed_count = st.slider("Number of seeds", 1, 5, 2)
    run = st.button("Run experiment", type="primary")

if run:
    seeds = list(range(1, seed_count + 1))
    if mode == "Compare both HMM profiles":
        results = profile_robustness_comparison(list_available_profiles(), scenarios, noise_rates, missing_rates, seeds)
    else:
        results = noise_robustness_experiment(profile, scenarios, noise_rates, missing_rates, seeds)
    st.session_state["robustness_results"] = results
else:
    results = st.session_state.get("robustness_results", pd.DataFrame())

if results.empty:
    panel(
        "Choose the experiment settings in the sidebar and click **Run experiment**. "
        "The lab intentionally does not run automatically on page load because Streamlit Cloud has limited shared CPU."
    )
    st.stop()

st.dataframe(results, hide_index=True, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(robustness_accuracy_chart(results, "noise_rate"), use_container_width=True)
with c2:
    st.plotly_chart(robustness_accuracy_chart(results, "missing_rate"), use_container_width=True)
st.plotly_chart(robustness_heatmap(results), use_container_width=True)

summary = results.groupby("profile", as_index=False)["phase_accuracy"].mean().sort_values("phase_accuracy", ascending=False)
best = summary.iloc[0]
high_noise = results[results["noise_rate"] == results["noise_rate"].max()]["phase_accuracy"].mean()
low_noise = results[results["noise_rate"] == results["noise_rate"].min()]["phase_accuracy"].mean()
panel(
    f"Best mean phase accuracy in this run: **{best['profile']}** at **{best['phase_accuracy']:.1%}**. "
    f"Average phase accuracy changed from **{low_noise:.1%}** at the lowest noise rate to **{high_noise:.1%}** at the highest configured noise rate."
)
