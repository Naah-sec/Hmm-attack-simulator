"""One-call analysis workflow used by Streamlit pages."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .explanation import generate_explanation
from .mitre_mapper import get_techniques_for_alert, rank_techniques_from_state_distribution, rank_techniques_from_next_distribution
from .model_profiles import load_profile


def analyze_alerts(profile_name: str, alerts: list[str], true_states: list[str] | None = None) -> dict[str, Any]:
    model = load_profile(profile_name)
    viterbi = model.viterbi(alerts)
    current = model.current_state_distribution(alerts)
    next_dist = model.predict_next_distribution(alerts)
    evolution = model.probability_evolution(alerts)
    mitre_current = rank_techniques_from_state_distribution(profile_name, current)
    mitre_next = rank_techniques_from_next_distribution(profile_name, next_dist)
    explanation = generate_explanation(model, alerts, viterbi["path"], current, next_dist, mitre_next or mitre_current)
    alert_mapping = {
        alert: ", ".join(item["technique_id"] for item in get_techniques_for_alert(alert)) or "None"
        for alert in sorted(set(alerts))
    }
    steps_df = pd.DataFrame(viterbi["steps"])
    if true_states:
        steps_df["true_state"] = true_states[: len(steps_df)]
    return {
        "model": model,
        "alerts": alerts,
        "true_states": true_states,
        "viterbi": viterbi,
        "current_distribution": current,
        "next_distribution": next_dist,
        "evolution": evolution,
        "mitre_current": mitre_current,
        "mitre_next": mitre_next,
        "mitre_techniques": mitre_next or mitre_current,
        "explanation": explanation,
        "alert_mapping": alert_mapping,
        "steps_df": steps_df,
    }

