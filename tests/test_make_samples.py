"""Tests for benchmark hydration and fixed-sample construction."""

import importlib.util
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "make_samples.py"
SPEC = importlib.util.spec_from_file_location("make_samples", SCRIPT)
make_samples = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(make_samples)


def test_parse_annotations_accepts_python_repr():
    value = "[{'label': 'EMAIL', 'text': 'a@b.co', 'start': 0, 'end': 6}]"
    assert make_samples._parse_annotations(value)[0]["label"] == "EMAIL"


def test_subsample_preserves_source_proportions_and_is_repeatable():
    rows = []
    for source, count in (("a", 60), ("b", 30), ("c", 10)):
        rows.extend(
            {
                "source": source,
                "text": f"row {i}",
                "ground_truth": "[]",
                "id": f"{source}-{i}",
            }
            for i in range(count)
        )

    first = make_samples.stratified_subsample(rows, 20, 42)
    second = make_samples.stratified_subsample(rows, 20, 42)

    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert Counter(row["source"] for row in first) == {"a": 12, "b": 6, "c": 2}
