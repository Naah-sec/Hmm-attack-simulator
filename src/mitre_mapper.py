"""MITRE ATT&CK enrichment helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config_loader import load_mitre_mapping


def _mapping() -> dict[str, Any]:
    return load_mitre_mapping()


def _technique_name(technique_id: str) -> str:
    return _mapping()["technique_names"].get(technique_id, "Unknown technique")


def get_techniques_for_state(profile_name: str, state: str) -> list[dict[str, str]]:
    techniques = _mapping()["profiles"].get(profile_name, {}).get(state, [])
    return [{"technique_id": tid, "technique_name": _technique_name(tid)} for tid in techniques]


def get_techniques_for_alert(alert: str) -> list[dict[str, str]]:
    techniques = _mapping()["alert_to_techniques"].get(alert, [])
    return [{"technique_id": tid, "technique_name": _technique_name(tid)} for tid in techniques]


def rank_techniques_from_state_distribution(profile_name: str, state_distribution: dict[str, float]) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    source: dict[str, tuple[str, float]] = {}
    for state, probability in state_distribution.items():
        techniques = _mapping()["profiles"].get(profile_name, {}).get(state, [])
        if not techniques:
            continue
        share = probability / len(techniques)
        for tid in techniques:
            scores[tid] += share
            if tid not in source or probability > source[tid][1]:
                source[tid] = (state, probability)
    ranked = []
    for tid, score in scores.items():
        state, confidence = source[tid]
        ranked.append(
            {
                "technique_id": tid,
                "technique_name": _technique_name(tid),
                "score": round(float(score * 100), 2),
                "source_state": state,
                "confidence": round(float(confidence), 4),
                "comment": f"Predicted from {state} phase.",
            }
        )
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def rank_techniques_from_next_distribution(profile_name: str, next_distribution: dict[str, float]) -> list[dict[str, Any]]:
    return rank_techniques_from_state_distribution(profile_name, next_distribution)


def build_navigator_technique_entries(technique_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "techniqueID": item["technique_id"],
            "score": int(round(float(item["score"]))),
            "comment": item.get("comment", ""),
        }
        for item in technique_scores
    ]

