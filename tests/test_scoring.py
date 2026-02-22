"""Tests for scoring metrics — precision, recall, F1 and span IoU."""

import pytest

from ablation_harness.scoring.metrics import (
    _span_int,
    compute_metrics,
    compute_span_iou,
)
from ablation_harness.scoring.normalize import normalize_pii_type

LABEL_MAP = {
    "PERSON": "PERSON",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "LOCATION": "LOCATION",
    "USERNAME": "PERSON",
    "EMAIL": "EMAIL_ADDRESS",
}


class TestComputeSpanIou:

    def test_perfect_overlap(self):
        assert compute_span_iou({"start": 0, "end": 10}, {"start": 0, "end": 10}) == 1.0

    def test_no_overlap(self):
        assert compute_span_iou({"start": 0, "end": 5}, {"start": 10, "end": 15}) == 0.0

    def test_partial_overlap(self):
        iou = compute_span_iou({"start": 0, "end": 10}, {"start": 5, "end": 15})
        assert 0.3 < iou < 0.4  # 5 / 15 ≈ 0.333

    def test_zero_length_spans(self):
        assert compute_span_iou({"start": 0, "end": 0}, {"start": 0, "end": 0}) == 0.0

    def test_redacted_start_end_treated_as_zero(self):
        """Spans with non-numeric start/end (e.g. '<REDACTED>') are coerced to 0; no ValueError."""
        redacted = {"start": "<REDACTED>", "end": "<REDACTED>"}
        normal = {"start": 0, "end": 10}
        assert compute_span_iou(redacted, normal) == 0.0
        assert compute_span_iou(normal, redacted) == 0.0
        assert compute_span_iou(redacted, redacted) == 0.0

    def test_string_numeric_start_end_coerced(self):
        """String numbers like '5' and '10' are coerced to int."""
        span1 = {"start": "0", "end": "10"}
        span2 = {"start": 0, "end": 10}
        assert compute_span_iou(span1, span2) == 1.0

    def test_missing_start_end_default_to_zero(self):
        assert compute_span_iou({}, {"start": 5, "end": 10}) == 0.0
        assert compute_span_iou({"type": "PERSON"}, {"start": 0, "end": 10}) == 0.0

    def test_none_start_end_treated_as_zero(self):
        assert (
            compute_span_iou({"start": None, "end": None}, {"start": 0, "end": 10})
            == 0.0
        )


class TestSpanInt:
    """Tests for _span_int coercion (avoids ValueError on redacted/invalid data)."""

    def test_valid_int(self):
        assert _span_int({"start": 5}, "start") == 5
        assert _span_int({"end": 10}, "end") == 10

    def test_missing_key_returns_zero(self):
        assert _span_int({}, "start") == 0
        assert _span_int({"type": "PERSON"}, "end") == 0

    def test_redacted_returns_zero(self):
        assert _span_int({"start": "<REDACTED>"}, "start") == 0
        assert _span_int({"end": "<REDACTED>"}, "end") == 0

    def test_string_number_coerced(self):
        assert _span_int({"start": "7"}, "start") == 7
        assert _span_int({"end": "14"}, "end") == 14

    def test_none_returns_zero(self):
        assert _span_int({"start": None}, "start") == 0
        assert _span_int({"end": None}, "end") == 0

    def test_invalid_string_returns_zero(self):
        assert _span_int({"start": "nope"}, "start") == 0
        assert _span_int({"end": ""}, "end") == 0


class TestNormalizePiiType:

    def test_direct_mapping(self):
        assert normalize_pii_type("PERSON", LABEL_MAP) == "PERSON"

    def test_alias_mapping(self):
        assert normalize_pii_type("USERNAME", LABEL_MAP) == "PERSON"
        assert normalize_pii_type("EMAIL", LABEL_MAP) == "EMAIL_ADDRESS"

    def test_case_normalization(self):
        assert normalize_pii_type("person", LABEL_MAP) == "PERSON"
        assert normalize_pii_type("email_address", LABEL_MAP) == "EMAIL_ADDRESS"

    def test_unknown_type(self):
        assert normalize_pii_type("", LABEL_MAP) == "UNKNOWN"

    def test_passthrough(self):
        assert normalize_pii_type("CUSTOM_TYPE", LABEL_MAP) == "CUSTOM_TYPE"


class TestComputeMetrics:

    def test_perfect_match_type_only(self):
        preds = [{"type": "PERSON"}, {"type": "EMAIL_ADDRESS"}]
        gt = [{"type": "PERSON"}, {"type": "EMAIL_ADDRESS"}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_no_predictions(self):
        m = compute_metrics([], [{"type": "PERSON"}], LABEL_MAP)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_no_ground_truth_no_preds(self):
        m = compute_metrics([], [], LABEL_MAP)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_no_ground_truth_with_preds(self):
        m = compute_metrics([{"type": "PERSON"}], [], LABEL_MAP)
        assert m["precision"] == 0.0
        assert m["recall"] == 1.0
        assert m["f1"] == 0.0

    def test_partial_match(self):
        preds = [{"type": "PERSON"}, {"type": "LOCATION"}]
        gt = [{"type": "PERSON"}, {"type": "EMAIL_ADDRESS"}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert m["precision"] == 0.5
        assert m["recall"] == 0.5

    def test_span_iou_matching(self):
        preds = [{"type": "PERSON", "start": 0, "end": 10}]
        gt = [{"type": "PERSON", "start": 0, "end": 10}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert m["f1"] == 1.0

    def test_span_iou_below_threshold(self):
        preds = [{"type": "PERSON", "start": 0, "end": 5}]
        gt = [{"type": "PERSON", "start": 10, "end": 20}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert m["f1"] == 0.0

    def test_alias_matching(self):
        """Predictions using alias labels should match ground truth."""
        preds = [{"type": "USERNAME"}]
        gt = [{"type": "PERSON"}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert m["f1"] == 1.0

    def test_redacted_span_in_prediction_no_value_error(self):
        """compute_metrics must not raise when prediction has redacted start/end."""
        preds = [{"type": "PERSON", "start": "<REDACTED>", "end": "<REDACTED>"}]
        gt = [{"type": "PERSON", "start": 0, "end": 10}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert "precision" in m and "recall" in m and "f1" in m
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_redacted_span_in_ground_truth_no_value_error(self):
        """compute_metrics must not raise when ground truth has redacted start/end."""
        preds = [{"type": "PERSON", "start": 0, "end": 10}]
        gt = [{"type": "PERSON", "start": "<REDACTED>", "end": "<REDACTED>"}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert "precision" in m and "recall" in m and "f1" in m
        assert m["f1"] == 1.0

    def test_redacted_both_sides_no_value_error(self):
        """Both pred and truth with redacted spans: no ValueError, type-only match."""
        preds = [{"type": "PERSON", "start": "<REDACTED>", "end": "<REDACTED>"}]
        gt = [{"type": "PERSON", "start": "<REDACTED>", "end": "<REDACTED>"}]
        m = compute_metrics(preds, gt, LABEL_MAP)
        assert m["f1"] == 1.0
