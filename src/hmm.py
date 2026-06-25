"""Manual Hidden Markov Model algorithms for defensive IDS-alert simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass
class HiddenMarkovModel:
    """A discrete-observation Hidden Markov Model with transparent inference APIs."""

    name: str
    description: str
    states: list[str]
    observations: list[str]
    initial_probs: np.ndarray
    transition_matrix: np.ndarray
    emission_matrix: np.ndarray
    state_to_index: dict[str, int] = field(init=False)
    observation_to_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.initial_probs = np.asarray(self.initial_probs, dtype=float)
        self.transition_matrix = np.asarray(self.transition_matrix, dtype=float)
        self.emission_matrix = np.asarray(self.emission_matrix, dtype=float)
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.observation_to_index = {obs: i for i, obs in enumerate(self.observations)}
        self.validate()

    def validate(self) -> None:
        n_states = len(self.states)
        n_observations = len(self.observations)
        if self.initial_probs.shape != (n_states,):
            raise ValueError(f"{self.name}: initial probability vector has invalid shape")
        if self.transition_matrix.shape != (n_states, n_states):
            raise ValueError(f"{self.name}: transition matrix must be {n_states}x{n_states}")
        if self.emission_matrix.shape != (n_states, n_observations):
            raise ValueError(f"{self.name}: emission matrix must be {n_states}x{n_observations}")
        if "UNKNOWN_ALERT" not in self.observation_to_index:
            raise ValueError(f"{self.name}: observations must include UNKNOWN_ALERT")
        for label, matrix in {
            "initial_probs": self.initial_probs.reshape(1, -1),
            "transition_matrix": self.transition_matrix,
            "emission_matrix": self.emission_matrix,
        }.items():
            if np.any(matrix < 0):
                raise ValueError(f"{self.name}: {label} contains negative probabilities")
        if not np.isclose(self.initial_probs.sum(), 1.0, atol=1e-6):
            raise ValueError(f"{self.name}: initial probabilities must sum to 1")
        for label, matrix in {
            "transition_matrix": self.transition_matrix,
            "emission_matrix": self.emission_matrix,
        }.items():
            rows = matrix.sum(axis=1)
            if not np.allclose(rows, 1.0, atol=1e-6):
                raise ValueError(f"{self.name}: {label} rows must sum to 1")

    def _obs(self, observation: str) -> str:
        return observation if observation in self.observation_to_index else "UNKNOWN_ALERT"

    def _obs_indices(self, observation_sequence: Sequence[str]) -> list[int]:
        if not observation_sequence:
            raise ValueError("Observation sequence must not be empty")
        return [self.observation_to_index[self._obs(obs)] for obs in observation_sequence]

    def _distribution_frame(self, distributions: np.ndarray, observations: Sequence[str]) -> pd.DataFrame:
        frame = pd.DataFrame(distributions, columns=self.states)
        frame.insert(0, "alert", [self._obs(obs) for obs in observations])
        frame.insert(0, "t", range(len(observations)))
        frame["most_likely_state"] = frame[self.states].idxmax(axis=1)
        frame["confidence"] = frame[self.states].max(axis=1)
        return frame

    def forward(self, observation_sequence: Sequence[str]) -> pd.DataFrame:
        """Return normalized forward posterior distributions after each observation."""
        indices = self._obs_indices(observation_sequence)
        alpha = self.initial_probs * self.emission_matrix[:, indices[0]]
        alpha = alpha / max(alpha.sum(), EPS)
        distributions = [alpha.copy()]
        for obs_idx in indices[1:]:
            alpha = (alpha @ self.transition_matrix) * self.emission_matrix[:, obs_idx]
            alpha = alpha / max(alpha.sum(), EPS)
            distributions.append(alpha.copy())
        return self._distribution_frame(np.vstack(distributions), observation_sequence)

    def sequence_log_likelihood(self, observation_sequence: Sequence[str]) -> float:
        """Compute scaled forward log likelihood for the observation sequence."""
        indices = self._obs_indices(observation_sequence)
        alpha = self.initial_probs * self.emission_matrix[:, indices[0]]
        scale = max(alpha.sum(), EPS)
        log_likelihood = float(np.log(scale))
        alpha = alpha / scale
        for obs_idx in indices[1:]:
            alpha = (alpha @ self.transition_matrix) * self.emission_matrix[:, obs_idx]
            scale = max(alpha.sum(), EPS)
            log_likelihood += float(np.log(scale))
            alpha = alpha / scale
        return log_likelihood

    def viterbi(self, observation_sequence: Sequence[str]) -> dict[str, object]:
        """Return the most likely hidden-state path using log probabilities."""
        indices = self._obs_indices(observation_sequence)
        n_states = len(self.states)
        log_initial = np.log(self.initial_probs + EPS)
        log_transition = np.log(self.transition_matrix + EPS)
        log_emission = np.log(self.emission_matrix + EPS)

        delta = np.zeros((len(indices), n_states), dtype=float)
        psi = np.zeros((len(indices), n_states), dtype=int)
        delta[0] = log_initial + log_emission[:, indices[0]]

        for t in range(1, len(indices)):
            scores = delta[t - 1][:, None] + log_transition
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_emission[:, indices[t]]

        path_idx = [int(np.argmax(delta[-1]))]
        for t in range(len(indices) - 1, 0, -1):
            path_idx.append(int(psi[t, path_idx[-1]]))
        path_idx.reverse()
        path = [self.states[i] for i in path_idx]

        forward_df = self.forward(observation_sequence)
        steps = []
        for t, (alert, state) in enumerate(zip(observation_sequence, path)):
            confidence = float(forward_df.loc[t, state])
            steps.append(
                {
                    "t": t,
                    "alert": self._obs(alert),
                    "predicted_state": state,
                    "confidence": confidence,
                }
            )
        return {
            "path": path,
            "path_probability_log": float(np.max(delta[-1])),
            "steps": steps,
        }

    def current_state_distribution(self, observation_sequence: Sequence[str]) -> dict[str, float]:
        row = self.forward(observation_sequence).iloc[-1]
        return {state: float(row[state]) for state in self.states}

    def predict_next_distribution(self, observation_sequence: Sequence[str]) -> dict[str, float]:
        current = np.array([self.current_state_distribution(observation_sequence)[s] for s in self.states])
        next_dist = current @ self.transition_matrix
        next_dist = next_dist / max(next_dist.sum(), EPS)
        return {state: float(next_dist[i]) for i, state in enumerate(self.states)}

    def predict_top_k_next_states(self, observation_sequence: Sequence[str], k: int = 3) -> list[dict[str, float | str]]:
        dist = self.predict_next_distribution(observation_sequence)
        return [
            {"state": state, "probability": float(prob)}
            for state, prob in sorted(dist.items(), key=lambda item: item[1], reverse=True)[:k]
        ]

    def probability_evolution(self, observation_sequence: Sequence[str]) -> pd.DataFrame:
        return self.forward(observation_sequence)

    def get_transition_probability(self, from_state: str, to_state: str) -> float:
        return float(self.transition_matrix[self.state_to_index[from_state], self.state_to_index[to_state]])

    def get_emission_probability(self, state: str, observation: str) -> float:
        obs = self._obs(observation)
        return float(self.emission_matrix[self.state_to_index[state], self.observation_to_index[obs]])

