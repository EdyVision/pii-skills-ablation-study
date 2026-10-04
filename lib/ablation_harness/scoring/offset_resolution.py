"""Deterministic span-text to character-offset resolution.

The resolver is deliberately simple and auditable. It performs exact,
case-sensitive substring matching and assigns repeated occurrences from left to
right, without consulting gold annotations or model-generated offsets.
"""

from __future__ import annotations

from collections import defaultdict


def _occurrences(source_text: str, span_text: str) -> list[tuple[int, int]]:
    """Return every possibly-overlapping exact occurrence of *span_text*."""
    if not span_text:
        return []

    matches = []
    start = 0
    while True:
        index = source_text.find(span_text, start)
        if index < 0:
            return matches
        matches.append((index, index + len(span_text)))
        start = index + 1


def resolve_prediction_offsets(
    source_text: str,
    predictions: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Resolve prediction text into offsets with a leftmost-unused policy.

    Returns resolved prediction copies and one privacy-safe diagnostic per
    input prediction. Unresolved predictions remain in the returned list with
    ``start`` and ``end`` set to ``None`` so they remain false positives in
    downstream scoring.
    """
    used_by_text: dict[str, set[tuple[int, int]]] = defaultdict(set)
    resolved = []
    diagnostics = []

    for index, prediction in enumerate(predictions or []):
        if not isinstance(prediction, dict):
            continue

        item = dict(prediction)
        span_text = item.get("text")
        span_text = span_text if isinstance(span_text, str) else ""
        candidates = _occurrences(source_text, span_text)
        available = [c for c in candidates if c not in used_by_text[span_text]]
        selected = available[0] if available else None

        native_start = item.get("start")
        native_end = item.get("end")
        native_matches_text = (
            isinstance(native_start, int)
            and isinstance(native_end, int)
            and 0 <= native_start <= native_end <= len(source_text)
            and source_text[native_start:native_end] == span_text
        )

        if selected is None:
            item["start"] = None
            item["end"] = None
            status = "missing_text" if not span_text else "unresolved"
        else:
            item["start"], item["end"] = selected
            used_by_text[span_text].add(selected)
            status = "resolved_unique" if len(candidates) == 1 else "resolved_repeated"

        resolved.append(item)
        diagnostics.append(
            {
                "prediction_index": index,
                "status": status,
                "candidate_count": len(candidates),
                "native_offsets_match_text": native_matches_text,
                "resolved_start": item.get("start"),
                "resolved_end": item.get("end"),
            }
        )

    return resolved, diagnostics
