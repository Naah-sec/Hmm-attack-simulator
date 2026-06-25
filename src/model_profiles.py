"""HMM profile loading and profile-to-profile state mapping."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config_loader import alert_vocabulary, load_json
from .hmm import HiddenMarkovModel


ENRICHED_PROFILE = "ATT&CK-Enriched HMM"
PUBLISHED_PROFILE = "Published APT-HMM"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Cannot normalize matrix with empty probability row")
    return arr / row_sums


def build_emission_matrix_from_weights(
    states: list[str],
    observations: list[str],
    emission_weights: dict[str, dict[str, float]],
    smoothing: float,
) -> np.ndarray:
    matrix = np.zeros((len(states), len(observations)), dtype=float)
    for i, state in enumerate(states):
        weights = emission_weights.get(state)
        if not weights:
            raise ValueError(f"Missing emission weights for state: {state}")
        wildcard = float(weights.get("*", smoothing))
        for j, observation in enumerate(observations):
            matrix[i, j] = float(weights.get(observation, wildcard))
    return normalize_rows(matrix)


def _load_profile(filename: str) -> HiddenMarkovModel:
    cfg = load_json(filename)
    observations = alert_vocabulary()
    states = list(cfg["states"])
    initial = np.array([cfg["initial_probs"][state] for state in states], dtype=float)
    initial = initial / initial.sum()
    transition = normalize_rows(np.array(cfg["transition_matrix"], dtype=float))
    emission = build_emission_matrix_from_weights(
        states=states,
        observations=observations,
        emission_weights=cfg["emission_weights"],
        smoothing=float(cfg.get("smoothing", 0.1)),
    )
    return HiddenMarkovModel(
        name=cfg["name"],
        description=cfg["description"],
        states=states,
        observations=observations,
        initial_probs=initial,
        transition_matrix=transition,
        emission_matrix=emission,
    )


@lru_cache(maxsize=2)
def load_attack_enriched_hmm() -> HiddenMarkovModel:
    return _load_profile("attack_enriched_hmm.json")


@lru_cache(maxsize=2)
def load_published_apt_hmm() -> HiddenMarkovModel:
    return _load_profile("published_apt_hmm.json")


def list_available_profiles() -> list[str]:
    return [ENRICHED_PROFILE, PUBLISHED_PROFILE]


def load_profile(profile_name: str) -> HiddenMarkovModel:
    if profile_name == ENRICHED_PROFILE:
        return load_attack_enriched_hmm()
    if profile_name == PUBLISHED_PROFILE:
        return load_published_apt_hmm()
    raise ValueError(f"Unknown profile: {profile_name}")


ENRICHED_TO_PUBLISHED = {
    "Reconnaissance": "S1_Point_of_Entry",
    "Initial Access": "S1_Point_of_Entry",
    "Execution": "S2_C2_Communications",
    "Privilege Escalation": "S2_C2_Communications",
    "Lateral Movement": "S3_Lateral_Movement",
    "Collection": "S4_Asset_Data_Discovery",
    "Exfiltration": "S5_Data_Exfiltration",
    "Noise": "S6_Non_Complete",
}


def map_state_between_profiles(state: str, source_profile: str, target_profile: str) -> str:
    if source_profile == target_profile:
        return state
    if source_profile == ENRICHED_PROFILE and target_profile == PUBLISHED_PROFILE:
        return ENRICHED_TO_PUBLISHED.get(state, "S6_Non_Complete")
    if source_profile == PUBLISHED_PROFILE and target_profile == ENRICHED_PROFILE:
        reverse = {
            "S1_Point_of_Entry": "Initial Access",
            "S2_C2_Communications": "Execution",
            "S3_Lateral_Movement": "Lateral Movement",
            "S4_Asset_Data_Discovery": "Collection",
            "S5_Data_Exfiltration": "Exfiltration",
            "S6_Non_Complete": "Noise",
        }
        return reverse.get(state, "Noise")
    raise ValueError(f"Unsupported profile mapping: {source_profile} -> {target_profile}")

