"""Tests for the equalized deterministic text-to-offset control."""

from ablation_harness.scoring.offset_resolution import resolve_prediction_offsets


def test_unique_exact_text_is_resolved_without_using_native_offsets():
    predictions = [{"type": "PERSON", "text": "Ada", "start": 99, "end": 102}]

    resolved, diagnostics = resolve_prediction_offsets("Hello Ada", predictions)

    assert (resolved[0]["start"], resolved[0]["end"]) == (6, 9)
    assert diagnostics[0]["status"] == "resolved_unique"
    assert diagnostics[0]["native_offsets_match_text"] is False


def test_repeated_text_uses_leftmost_unused_occurrences():
    predictions = [
        {"type": "PERSON", "text": "Ada"},
        {"type": "PERSON", "text": "Ada"},
        {"type": "PERSON", "text": "Ada"},
    ]

    resolved, diagnostics = resolve_prediction_offsets("Ada met Ada", predictions)

    assert [(x["start"], x["end"]) for x in resolved] == [
        (0, 3),
        (8, 11),
        (None, None),
    ]
    assert [x["status"] for x in diagnostics] == [
        "resolved_repeated",
        "resolved_repeated",
        "unresolved",
    ]


def test_missing_or_non_string_text_stays_unresolved():
    resolved, diagnostics = resolve_prediction_offsets(
        "Ada", [{"type": "PERSON"}, {"type": "ID", "text": 123}]
    )

    assert all(x["start"] is None and x["end"] is None for x in resolved)
    assert all(x["status"] == "missing_text" for x in diagnostics)


def test_resolution_is_case_sensitive_and_supports_unicode():
    resolved, _ = resolve_prediction_offsets(
        "José met JOSE", [{"type": "PERSON", "text": "José"}]
    )

    assert (resolved[0]["start"], resolved[0]["end"]) == (0, 4)
