"""Analyst-friendly explanations for HMM predictions."""

from __future__ import annotations

from typing import Any

from .hmm import HiddenMarkovModel


def confidence_label(confidence: float) -> str:
    if confidence >= 0.70:
        return "High"
    if confidence >= 0.40:
        return "Medium"
    return "Low"


def generate_explanation(
    model: HiddenMarkovModel,
    observations: list[str],
    viterbi_path: list[str],
    current_distribution: dict[str, float],
    next_distribution: dict[str, float],
    mitre_techniques: list[dict[str, Any]],
) -> dict[str, Any]:
    current_state, current_conf = max(current_distribution.items(), key=lambda item: item[1])
    next_state, next_prob = max(next_distribution.items(), key=lambda item: item[1])
    recent = observations[-3:]
    emission_details = [
        {
            "alert": alert,
            "state": current_state,
            "emission_probability": model.get_emission_probability(current_state, alert),
        }
        for alert in recent
    ]
    important = sorted(emission_details, key=lambda item: item["emission_probability"], reverse=True)[:2]
    important_alerts = ", ".join(item["alert"] for item in important) or "the latest observations"
    transition_prob = model.get_transition_probability(current_state, next_state)
    top_techniques = ", ".join(
        f"{item['technique_id']} {item['technique_name']}" for item in mitre_techniques[:4]
    ) or "no direct ATT&CK techniques"
    caveats = []
    if any(alert in {"NORMAL_TRAFFIC", "UNKNOWN_ALERT"} for alert in observations):
        caveats.append("Noise or unknown alerts influenced the posterior distribution.")
    if current_conf < 0.40:
        caveats.append("The current phase has low confidence; interpret ranking rather than a hard label.")
    if model.name == "Published APT-HMM" and current_state in {"S5_Data_Exfiltration", "S6_Non_Complete"}:
        caveats.append(
            "Transitions back to S1 in this published profile represent new-campaign or scenario-restart tendency."
        )

    label = confidence_label(current_conf)
    analyst = (
        f"The current inferred phase is {current_state} with {current_conf:.0%} confidence ({label}). "
        f"This is mainly supported by {important_alerts}, which are plausible emissions under {current_state}. "
        f"The model predicts {next_state} as the next most likely phase because the transition "
        f"{current_state} -> {next_state} has probability {transition_prob:.2f}. "
        f"The most relevant MITRE ATT&CK techniques are {top_techniques}."
    )
    if caveats:
        analyst += " Caveat: " + " ".join(caveats)

    executive = (
        f"{model.name} infers {current_state} as the active phase and ranks {next_state} as the most likely next step. "
        f"The decision combines recent IDS alert emissions with campaign-stage transition probabilities and maps the result "
        f"to ATT&CK techniques for analyst review."
    )
    math = (
        "The HMM estimates P(state_t | observations_1:t) with the forward algorithm and finds the most likely hidden "
        "state path with Viterbi. Each step combines transition probability P(state_t | state_t-1) and emission "
        "probability P(alert_t | state_t), then normalizes the posterior distribution."
    )
    return {
        "current_phase": current_state,
        "current_confidence": current_conf,
        "confidence_label": label,
        "next_phase": next_state,
        "next_probability": next_prob,
        "transition_probability": transition_prob,
        "recent_alerts": recent,
        "emission_details": emission_details,
        "analyst_explanation": analyst,
        "executive_summary": executive,
        "mathematical_explanation": math,
        "caveats": caveats,
        "viterbi_terminal_state": viterbi_path[-1] if viterbi_path else current_state,
    }

