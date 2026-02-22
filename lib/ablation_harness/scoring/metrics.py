"""Scoring metrics — precision, recall, F1 with optional span IoU."""

from __future__ import annotations

import json

from ablation_harness.scoring.normalize import normalize_pii_type


def _span_int(d: dict, key: str) -> int:
    """Coerce span start/end to int; use 0 for missing or non-numeric (e.g. '<REDACTED>')."""
    val = d.get(key, 0)
    if isinstance(val, int):
        return val
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def compute_span_iou(span1: dict, span2: dict) -> float:
    """Compute Intersection over Union for two spans."""
    start1 = _span_int(span1, "start")
    end1 = _span_int(span1, "end")
    start2 = _span_int(span2, "start")
    end2 = _span_int(span2, "end")

    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = (end1 - start1) + (end2 - start2) - intersection

    return intersection / union if union > 0 else 0.0


def compute_metrics(
    predictions: list,
    ground_truth: list,
    label_map: dict,
    overlap_threshold: float = 0.5,
) -> dict:
    """Compute precision, recall, F1 for PII detection."""
    valid_preds = [p for p in predictions if isinstance(p, dict) and "type" in p]

    if not ground_truth:
        return {
            "precision": 1.0 if not valid_preds else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not valid_preds else 0.0,
        }

    if not valid_preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matched_truth: set[int] = set()
    tp = 0

    for pred in valid_preds:
        pred_type = normalize_pii_type(pred.get("type", ""), label_map)

        for i, truth in enumerate(ground_truth):
            if i in matched_truth:
                continue

            truth_type = normalize_pii_type(truth.get("type", ""), label_map)
            if pred_type != truth_type:
                continue

            truth_has_span = (
                _span_int(truth, "start") != 0 or _span_int(truth, "end") != 0
            )
            if not truth_has_span:
                tp += 1
                matched_truth.add(i)
                break
            iou = compute_span_iou(pred, truth)
            if iou >= overlap_threshold:
                tp += 1
                matched_truth.add(i)
                break

    precision = tp / len(valid_preds) if valid_preds else 0
    recall = tp / len(ground_truth) if ground_truth else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def ground_truth_for_scoring(sample: dict) -> list[dict]:
    """Use hydrated ground_truth (with spans) when available; else type-only."""
    try:
        gt_raw = (
            json.loads(sample["ground_truth"]) if sample.get("ground_truth") else []
        )
    except (TypeError, json.JSONDecodeError):
        gt_raw = []
    has_spans = any(
        _span_int(g, "start") != 0 or _span_int(g, "end") != 0 for g in gt_raw
    )
    if has_spans:
        return gt_raw
    if sample.get("pii_codex_ground_truth"):
        try:
            return [{"type": t} for t in json.loads(sample["pii_codex_ground_truth"])]
        except (TypeError, json.JSONDecodeError):
            pass
    return gt_raw
