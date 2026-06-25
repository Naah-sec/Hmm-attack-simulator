"""Synthetic defensive IDS alert scenario simulation utilities."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import numpy as np

from .config_loader import alert_vocabulary, load_json
from .model_profiles import ENRICHED_PROFILE, PUBLISHED_PROFILE, map_state_between_profiles


def load_scenarios() -> dict[str, dict[str, Any]]:
    return load_json("scenarios.json")


def get_scenario_names() -> list[str]:
    return list(load_scenarios().keys())


def get_scenario(name: str) -> dict[str, Any]:
    scenarios = load_scenarios()
    if name not in scenarios:
        raise ValueError(f"Unknown scenario: {name}")
    return deepcopy(scenarios[name])


def _noise_state(true_states: list[str]) -> str:
    return "S6_Non_Complete" if any(state.startswith("S") for state in true_states) else "Noise"


def inject_noise(alerts: list[str], true_states: list[str], noise_rate: float, seed: int) -> tuple[list[str], list[str]]:
    if len(alerts) != len(true_states):
        raise ValueError("alerts and true_states must have the same length")
    rng = np.random.default_rng(seed)
    new_alerts = list(alerts)
    new_states = list(true_states)
    insertions = int(round(len(alerts) * max(noise_rate, 0.0)))
    state = _noise_state(true_states)
    for _ in range(insertions):
        position = int(rng.integers(0, len(new_alerts) + 1))
        new_alerts.insert(position, str(rng.choice(["NORMAL_TRAFFIC", "UNKNOWN_ALERT"])))
        new_states.insert(position, state)
    return new_alerts, new_states


def remove_missing_alerts(alerts: list[str], true_states: list[str], missing_rate: float, seed: int) -> tuple[list[str], list[str]]:
    if len(alerts) != len(true_states):
        raise ValueError("alerts and true_states must have the same length")
    rng = np.random.default_rng(seed)
    keep = rng.random(len(alerts)) >= max(missing_rate, 0.0)
    if not keep.any():
        keep[int(rng.integers(0, len(alerts)))] = True
    return [a for a, k in zip(alerts, keep) if k], [s for s, k in zip(true_states, keep) if k]


def _states_for_profile(states: list[str], profile_name: str) -> list[str]:
    if profile_name == ENRICHED_PROFILE:
        return states
    if profile_name == PUBLISHED_PROFILE:
        return [map_state_between_profiles(state, ENRICHED_PROFILE, PUBLISHED_PROFILE) for state in states]
    raise ValueError(f"Unknown profile: {profile_name}")


def simulate_scenario(
    name: str,
    profile_name: str,
    noise_rate: float = 0.0,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    scenario = get_scenario(name)
    alerts = list(scenario["alerts"])
    true_states = _states_for_profile(list(scenario["true_states"]), profile_name)
    if noise_rate > 0:
        alerts, true_states = inject_noise(alerts, true_states, noise_rate, seed)
    if missing_rate > 0:
        alerts, true_states = remove_missing_alerts(alerts, true_states, missing_rate, seed + 1009)
    return {
        "scenario_name": name,
        "description": scenario.get("description", ""),
        "alerts": alerts,
        "true_states": true_states,
        "profile_name": profile_name,
        "noise_rate": noise_rate,
        "missing_rate": missing_rate,
        "seed": seed,
    }


def custom_sequence_from_user_input(raw_text: str) -> tuple[list[str], list[str]]:
    vocabulary = set(alert_vocabulary())
    tokens = [token.strip().upper() for token in re.split(r"[\n,]+", raw_text or "") if token.strip()]
    alerts: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        if token in vocabulary:
            alerts.append(token)
        else:
            alerts.append("UNKNOWN_ALERT")
            unknown.append(token)
    if not alerts:
        alerts = ["UNKNOWN_ALERT"]
    return alerts, unknown

