"""Tests for privacy-safe result persistence and run provenance."""

import json

from ablation_harness.config import HarnessConfig
from ablation_harness.runner import save_results_to_disk


def test_save_redacts_text_hashes_response_and_writes_manifest(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "zero_shot.txt").write_text("Find PII in {text}")
    config = HarnessConfig(
        results_dir=tmp_path / "results",
        prompts_dir=prompts,
        conditions=["zero_shot"],
        run_id="test-seed42",
    )
    results = [
        {
            "predictions": [{"text": "secret", "start": 0, "end": 6}],
            "resolved_predictions": [{"text": "secret", "start": 0, "end": 6}],
            "ground_truth": [{"text": "secret", "start": 0, "end": 6}],
            "raw_response": "model said secret",
        }
    ]

    path = save_results_to_disk(config, results)
    saved = json.loads(path.read_text())[0]
    manifest = json.loads((config.results_dir / "run_manifest.json").read_text())

    assert saved["predictions"][0]["text"] == "[REDACTED]"
    assert saved["resolved_predictions"][0]["text"] == "[REDACTED]"
    assert saved["ground_truth"][0]["text"] == "[REDACTED]"
    assert "raw_response" not in saved
    assert len(saved["raw_response_sha256"]) == 64
    assert manifest["run_id"] == "test-seed42"
    assert manifest["prompt_sha256"]["zero_shot"]
