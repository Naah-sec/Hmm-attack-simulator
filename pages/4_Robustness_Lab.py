from __future__ import annotations

import pandas as pd
import streamlit as st

from src.evaluator import noise_robustness_experiment, profile_robustness_comparison
from src.model_profiles import list_available_profiles
from src.simulator import get_scenario_names
from src.utils import inject_css, metric_card, panel
from src.visualization import robustness_accuracy_chart, robustness_heatmap, robustness_profile_summary_chart


st.set_page_config(page_title="Robustness Lab", layout="wide")
inject_css()
st.title("Robustness Lab")
st.caption("Stress-test the HMM profiles against noisy IDS streams, missing alerts, and longer campaign scenarios.")


@st.cache_data(show_spinner=False)
def run_robustness_experiment(
    mode: str,
    profile: str,
    scenarios: tuple[str, ...],
    noise_rates: tuple[float, ...],
    missing_rates: tuple[float, ...],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    if mode == "Compare both HMM profiles":
        return profile_robustness_comparison(list_available_profiles(), list(scenarios), list(noise_rates), list(missing_rates), list(seeds))
    return noise_robustness_experiment(profile, list(scenarios), list(noise_rates), list(missing_rates), list(seeds))


with st.sidebar:
    mode = st.radio("Experiment mode", ["Single profile", "Compare both HMM profiles"])
    profile = st.selectbox("Profile", list_available_profiles(), disabled=mode != "Single profile")
    all_scenarios = get_scenario_names()
    default_scenarios = [name for name in all_scenarios if name.startswith("Advanced -")][:2] or all_scenarios[:3]
    scenarios = st.multiselect("Scenarios", all_scenarios, default=default_scenarios)
    noise_rates = st.multiselect("Noise rates", [0.0, 0.1, 0.2, 0.3, 0.4], default=[0.0, 0.2, 0.4])
    missing_rates = st.multiselect("Missing-alert rates", [0.0, 0.1, 0.2, 0.3], default=[0.0, 0.2])
    seed_count = st.slider("Number of seeds", 1, 5, 3)
    run = st.button("Run experiment", type="primary")

if run:
    if not scenarios or not noise_rates or not missing_rates:
        st.error("Select at least one scenario, one noise rate, and one missing-alert rate.")
        st.stop()
    with st.spinner("Running HMM robustness experiment..."):
        results = run_robustness_experiment(
            mode,
            profile,
            tuple(scenarios),
            tuple(noise_rates),
            tuple(missing_rates),
            tuple(range(1, seed_count + 1)),
        )
    st.session_state["robustness_results"] = results
else:
    results = st.session_state.get("robustness_results", pd.DataFrame())

if results.empty:
    panel(
        "Choose experiment settings in the sidebar and click **Run experiment**. "
        "A realistic demo is: compare both profiles, choose two advanced scenarios, use noise rates 0.0/0.2/0.4, "
        "missing-alert rates 0.0/0.2, and 3 seeds."
    )
    st.stop()

mean_phase = float(results["phase_accuracy"].mean())
mean_next = float(results["next_step_accuracy"].mean())
mean_top3 = float(results["top3_next_step_accuracy"].mean())
mean_confidence = float(results["avg_current_confidence"].mean())
mean_stress = float(results["stress_score"].mean())
run_count = len(results)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Experiment runs", f"{run_count}", "Profile x scenario x stress x seed")
with c2:
    metric_card("Phase accuracy", f"{mean_phase:.0%}", "Exact hidden phase match")
with c3:
    metric_card("Next-step accuracy", f"{mean_next:.0%}", "Top next state equals true next phase")
with c4:
    metric_card("Top-3 next step", f"{mean_top3:.0%}", "True next phase appears in top 3")
with c5:
    metric_card("Mean stress", f"{mean_stress:.0f}/100", "Combined noise and missing-alert pressure")

summary = results.groupby("profile", as_index=False).agg(
    phase_accuracy=("phase_accuracy", "mean"),
    next_step_accuracy=("next_step_accuracy", "mean"),
    top3_next_step_accuracy=("top3_next_step_accuracy", "mean"),
    avg_current_confidence=("avg_current_confidence", "mean"),
    mean_log_likelihood_per_alert=("mean_log_likelihood_per_alert", "mean"),
)
best = summary.sort_values("phase_accuracy", ascending=False).iloc[0]

high_noise = results[results["noise_rate"] == results["noise_rate"].max()]["phase_accuracy"].mean()
low_noise = results[results["noise_rate"] == results["noise_rate"].min()]["phase_accuracy"].mean()
degradation = low_noise - high_noise
panel(
    f"Best profile in this run: **{best['profile']}** with **{best['phase_accuracy']:.1%}** mean phase accuracy. "
    f"Accuracy changed from **{low_noise:.1%}** at the lowest configured noise rate to **{high_noise:.1%}** "
    f"at the highest configured noise rate, a degradation of **{degradation:.1%}**."
)

tabs = st.tabs(["Executive View", "Detailed Results", "Stress Charts", "Metric Guide"])

with tabs[0]:
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(robustness_profile_summary_chart(results), use_container_width=True)
    with right:
        scenario_summary = (
            results.groupby("scenario", as_index=False)
            .agg(
                phase_accuracy=("phase_accuracy", "mean"),
                top3_next_step_accuracy=("top3_next_step_accuracy", "mean"),
                stress_score=("stress_score", "mean"),
                observed_alert_count=("observed_alert_count", "mean"),
            )
            .sort_values("phase_accuracy")
        )
        st.subheader("Most Challenging Scenarios")
        st.dataframe(
            scenario_summary,
            hide_index=True,
            use_container_width=True,
            column_config={
                "phase_accuracy": st.column_config.ProgressColumn("Phase accuracy", format="%.1f", min_value=0, max_value=1),
                "top3_next_step_accuracy": st.column_config.ProgressColumn("Top-3 next step", format="%.1f", min_value=0, max_value=1),
                "stress_score": st.column_config.NumberColumn("Stress score", format="%.0f"),
                "observed_alert_count": st.column_config.NumberColumn("Avg alerts", format="%.1f"),
            },
        )

with tabs[1]:
    st.subheader("Experiment Result Rows")
    st.dataframe(
        results.sort_values(["profile", "scenario", "noise_rate", "missing_rate", "seed"]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "noise_rate": st.column_config.NumberColumn("Noise rate", format="%.1f"),
            "missing_rate": st.column_config.NumberColumn("Missing rate", format="%.1f"),
            "stress_score": st.column_config.NumberColumn("Stress score", format="%.0f"),
            "phase_accuracy": st.column_config.ProgressColumn("Phase accuracy", format="%.2f", min_value=0, max_value=1),
            "next_step_accuracy": st.column_config.ProgressColumn("Next-step accuracy", format="%.2f", min_value=0, max_value=1),
            "top3_next_step_accuracy": st.column_config.ProgressColumn("Top-3 next step", format="%.2f", min_value=0, max_value=1),
            "avg_current_confidence": st.column_config.ProgressColumn("Current confidence", format="%.2f", min_value=0, max_value=1),
            "sequence_log_likelihood": st.column_config.NumberColumn("Seq log likelihood", format="%.2f"),
            "mean_log_likelihood_per_alert": st.column_config.NumberColumn("Mean log likelihood / alert", format="%.2f"),
        },
    )

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(robustness_accuracy_chart(results, "noise_rate", "phase_accuracy"), use_container_width=True)
    with c2:
        st.plotly_chart(robustness_accuracy_chart(results, "missing_rate", "phase_accuracy"), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(robustness_accuracy_chart(results, "noise_rate", "next_step_accuracy"), use_container_width=True)
    with c4:
        st.plotly_chart(robustness_accuracy_chart(results, "noise_rate", "avg_current_confidence"), use_container_width=True)
    st.plotly_chart(robustness_heatmap(results), use_container_width=True)

with tabs[3]:
    st.subheader("What Each Value Means")
    metric_rows = [
        {
            "value": "profile",
            "meaning": "The HMM profile evaluated in that row.",
            "calculation": "Selected profile: ATT&CK-Enriched HMM or Published APT-HMM.",
        },
        {
            "value": "scenario",
            "meaning": "The synthetic IDS campaign sequence used for the run.",
            "calculation": "Loaded from data/scenarios.json, then perturbed by noise and missing-alert settings.",
        },
        {
            "value": "noise_rate",
            "meaning": "How many extra benign or unknown alerts are inserted into the sequence.",
            "calculation": "round(original_alert_count x noise_rate) inserted alerts. Inserted labels are NORMAL_TRAFFIC or UNKNOWN_ALERT.",
        },
        {
            "value": "missing_rate",
            "meaning": "How aggressively existing alert/state pairs are removed to simulate sensor loss or dropped telemetry.",
            "calculation": "Each original alert is independently kept with probability 1 - missing_rate. At least one alert is always kept.",
        },
        {
            "value": "seed",
            "meaning": "Random seed controlling where noise is inserted and which alerts are removed.",
            "calculation": "The same seed makes the same perturbed sequence reproducible.",
        },
        {
            "value": "observed_alert_count",
            "meaning": "Final alert count after noise insertion and missing-alert removal.",
            "calculation": "len(alerts) after perturbation.",
        },
        {
            "value": "stress_score",
            "meaning": "A readable 0-100 pressure score combining inserted noise and missing telemetry.",
            "calculation": "100 x (1 - ((1 - noise_rate) x (1 - missing_rate))). Higher means a more degraded alert stream.",
        },
        {
            "value": "phase_accuracy",
            "meaning": "How often Viterbi predicted exactly the same hidden phase as the scenario label.",
            "calculation": "correct_predicted_states / aligned_time_steps.",
        },
        {
            "value": "next_step_accuracy",
            "meaning": "How often the model's single most likely next phase matched the next true phase.",
            "calculation": "For each prefix alerts[0:t], compute predict_next_distribution; count a hit if argmax equals true_state[t+1].",
        },
        {
            "value": "top3_next_step_accuracy",
            "meaning": "How often the true next phase appeared anywhere in the model's top three predicted next phases.",
            "calculation": "Same next-step test, but a hit is counted if true_state[t+1] is in the top 3 states by probability.",
        },
        {
            "value": "avg_current_confidence",
            "meaning": "The final posterior confidence of the most likely current state.",
            "calculation": "max(current_state_distribution) after the full alert sequence.",
        },
        {
            "value": "sequence_log_likelihood",
            "meaning": "How probable the complete observed alert sequence is under the selected HMM.",
            "calculation": "Scaled forward algorithm log P(observations | model). Values are negative; less negative is generally better for similar-length sequences.",
        },
        {
            "value": "mean_log_likelihood_per_alert",
            "meaning": "Length-normalized likelihood, easier to compare across short and long scenarios.",
            "calculation": "sequence_log_likelihood / observed_alert_count. Less negative indicates the model explains each alert better on average.",
        },
    ]
    st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
    panel(
        "Interpretation rule: accuracy metrics measure correctness against the scenario labels, confidence measures how concentrated "
        "the model's final belief is, and log likelihood measures how well the model explains the observed alert sequence. "
        "A confident model can still be wrong, so confidence should be read beside phase accuracy."
    )
