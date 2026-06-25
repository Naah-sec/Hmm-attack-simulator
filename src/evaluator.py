"""Robustness and accuracy metrics for the two HMM profiles."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .export import export_experiment_results_csv
from .model_profiles import load_profile
from .simulator import get_scenario_names, simulate_scenario


def phase_accuracy(predicted_path: list[str], true_states: list[str]) -> float:
    n = min(len(predicted_path), len(true_states))
    if n == 0:
        return 0.0
    return sum(predicted_path[i] == true_states[i] for i in range(n)) / n


def current_state_confidence(current_distribution: dict[str, float]) -> float:
    return max(current_distribution.values()) if current_distribution else 0.0


def next_step_accuracy(model, alerts: list[str], true_states: list[str]) -> float:
    correct = 0
    total = 0
    for i in range(len(alerts) - 1):
        predicted = max(model.predict_next_distribution(alerts[: i + 1]).items(), key=lambda item: item[1])[0]
        correct += predicted == true_states[i + 1]
        total += 1
    return correct / total if total else 0.0


def top_k_next_step_accuracy(model, alerts: list[str], true_states: list[str], k: int = 3) -> float:
    correct = 0
    total = 0
    for i in range(len(alerts) - 1):
        top = {item["state"] for item in model.predict_top_k_next_states(alerts[: i + 1], k=k)}
        correct += true_states[i + 1] in top
        total += 1
    return correct / total if total else 0.0


def average_sequence_log_likelihood(model, sequences: Iterable[list[str]]) -> float:
    values = [model.sequence_log_likelihood(seq) / max(len(seq), 1) for seq in sequences if seq]
    return float(np.mean(values)) if values else 0.0


def noise_robustness_experiment(
    profile_name: str,
    scenarios: list[str] | None = None,
    noise_rates: list[float] | None = None,
    missing_rates: list[float] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    scenarios = scenarios or get_scenario_names()
    noise_rates = noise_rates or [0.0, 0.1, 0.2, 0.3, 0.4]
    missing_rates = missing_rates or [0.0, 0.1, 0.2, 0.3]
    seeds = seeds or [1, 2, 3, 4, 5]
    model = load_profile(profile_name)
    rows = []
    for scenario in scenarios:
        for noise_rate in noise_rates:
            for missing_rate in missing_rates:
                for seed in seeds:
                    simulation = simulate_scenario(scenario, profile_name, noise_rate, missing_rate, seed)
                    alerts = simulation["alerts"]
                    true_states = simulation["true_states"]
                    vit = model.viterbi(alerts)
                    current = model.current_state_distribution(alerts)
                    rows.append(
                        {
                            "profile": profile_name,
                            "scenario": scenario,
                            "noise_rate": noise_rate,
                            "missing_rate": missing_rate,
                            "seed": seed,
                            "phase_accuracy": phase_accuracy(vit["path"], true_states),
                            "next_step_accuracy": next_step_accuracy(model, alerts, true_states),
                            "top3_next_step_accuracy": top_k_next_step_accuracy(model, alerts, true_states, k=3),
                            "avg_current_confidence": current_state_confidence(current),
                            "sequence_log_likelihood": model.sequence_log_likelihood(alerts),
                        }
                    )
    results = pd.DataFrame(rows)
    export_experiment_results_csv(results)
    return results


def profile_robustness_comparison(
    profiles: list[str],
    scenarios: list[str] | None = None,
    noise_rates: list[float] | None = None,
    missing_rates: list[float] | None = None,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    frames = [
        noise_robustness_experiment(profile, scenarios, noise_rates, missing_rates, seeds)
        for profile in profiles
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

