#!/usr/bin/env python3
"""
Upload results from results/past_runs/<run_type>/ (e.g. main, pilot, rr) to the Hub.

Reads experiment_results.json, builds records with a fixed schema so Hub features
match across splits (scores and error always string to avoid Value('null') inference).
Data is unchanged; only column types are normalized for upload.

Usage:
  python scripts/upload_results_to_hub.py [--run-type main] [--prompt-version v1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _results_to_records(results_list: list[dict], run_type: str, pv: str) -> list[dict]:
    """Build Hub records with fixed schema: scores and error always string so features match across splits."""
    records = []
    for r in results_list:
        scores = r.get("scores")
        if scores is None:
            scores_str = "[]"
        elif isinstance(scores, str):
            scores_str = scores
        else:
            scores_str = json.dumps(scores)
        err = r.get("error")
        error_str = (
            "" if err is None else (err if isinstance(err, str) else json.dumps(err))
        )
        records.append(
            {
                "sample_id": r.get("sample_id"),
                "source": r.get("source"),
                "model": r.get("model"),
                "condition": r.get("condition"),
                "prompt_version": pv,
                "run_type": run_type,
                "predictions": json.dumps(r.get("predictions") or []),
                "scores": scores_str,
                "error": error_str,
                "tool_executed": bool(r.get("tool_executed", False)),
                "skill_viewed": bool(r.get("skill_viewed", False)),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "conversation_turns": r.get("conversation_turns"),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload results from results/<run_type>/ to Hub (fixed schema, no data change)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repo root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        default="main",
        help="Run type subdir under results/ (default: main).",
    )
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="Hugging Face dataset repo (default: from config.yaml harness.*.hub_repo).",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v1",
        help="Prompt version for split name (default: v1).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print counts only; do not push.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    run_type = args.run_type
    pv = args.prompt_version

    if args.hub_repo is not None:
        hub_repo = args.hub_repo.strip()
    else:
        import yaml

        config_path = repo_root / "config.yaml"
        if not config_path.exists():
            raise SystemExit("config.yaml not found; pass --hub-repo.")
        with open(config_path) as f:
            data = yaml.safe_load(f)
        harness = data.get("harness") or {}
        common = harness.get("common") or {}
        mlx = harness.get("mlx") or {}
        cuda = harness.get("cuda") or {}
        hub_repo = (
            mlx.get("hub_repo") or cuda.get("hub_repo") or common.get("hub_repo") or ""
        ).strip()
        if not hub_repo:
            raise SystemExit("hub_repo not in config; pass --hub-repo.")

    try:
        from datasets import Dataset, Features, Value
    except ImportError:
        raise SystemExit("pip install datasets")

    # Runs are stored under results/past_runs/<run_type>/ (e.g. main, pilot, rr).
    # Fall back to the legacy results/<run_type>/ location if past_runs is absent.
    exp_path = repo_root / "results" / "past_runs" / run_type / "experiment_results.json"
    if not exp_path.exists():
        legacy = repo_root / "results" / run_type / "experiment_results.json"
        if legacy.exists():
            exp_path = legacy
        else:
            raise SystemExit(
                f"Not found: {exp_path} (use --run-type for a different subdir)."
            )
    with open(exp_path) as f:
        results_list = json.load(f)
    if not results_list:
        raise SystemExit(f"Empty list in {exp_path}.")
    records = _results_to_records(results_list, run_type=run_type, pv=pv)
    split_name = f"{run_type}_{pv}"
    print(f"  {run_type}: {len(records)} results -> split {split_name}")

    if args.dry_run:
        print("Dry run: not pushing.")
        return

    hub_features = Features(
        {
            "sample_id": Value("string"),
            "source": Value("string"),
            "model": Value("string"),
            "condition": Value("string"),
            "prompt_version": Value("string"),
            "run_type": Value("string"),
            "predictions": Value("string"),
            "scores": Value("string"),
            "error": Value("string"),
            "tool_executed": Value("bool"),
            "skill_viewed": Value("bool"),
            "elapsed_seconds": Value("float64"),
            "conversation_turns": Value("int64"),
        }
    )
    ds = Dataset.from_list(records, features=hub_features)
    ds.push_to_hub(
        hub_repo,
        split=split_name,
        private=True,
        commit_message=f"Upload {split_name}: {len(records)} results.",
    )
    print(f"  Pushed {hub_repo} split={split_name} n={len(records)}")


if __name__ == "__main__":
    main()
