"""Configuration loading helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_json(filename: str) -> dict[str, Any]:
    """Load a JSON file from the project data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=8)
def load_alert_catalog() -> dict[str, dict[str, Any]]:
    return load_json("alert_catalog.json")


@lru_cache(maxsize=8)
def load_mitre_mapping() -> dict[str, Any]:
    return load_json("mitre_mapping.json")


def alert_vocabulary() -> list[str]:
    return list(load_alert_catalog().keys())

