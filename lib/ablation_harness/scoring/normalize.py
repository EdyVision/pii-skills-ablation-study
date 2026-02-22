"""PII type normalization with a configurable label map."""

from __future__ import annotations

import json
from pathlib import Path


def normalize_pii_type(pii_type: str, label_map: dict) -> str:
    """Normalize prediction labels to canonical PII types using *label_map*."""
    if not pii_type:
        return "UNKNOWN"
    normalized = pii_type.upper().replace("-", "_").replace(" ", "_")
    out = label_map.get(normalized, normalized)
    return out if out is not None else "UNKNOWN"


def load_label_map(path: str | Path) -> dict:
    """Load a label map from a JSON file."""
    with open(path) as f:
        return json.load(f)
