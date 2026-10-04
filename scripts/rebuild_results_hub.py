#!/usr/bin/env python3
"""
Rebuild the entire results dataset on the Hub with one uniform schema.

Why this exists:
  The original splits (pilot_v1, main_v1) were uploaded with error: null, but the
  current pipeline writes error as a string. Pushing a string-error split next to
  a null-error split fails with:
    "Features of the new split don't match the features of the existing splits".
  Pushing split-by-split can't fix it, because each new push is compared against
  the null-error splits still on the Hub.

What this does:
  Clears the existing data/ folder and README, then pushes ALL local run types as
  a single DatasetDict with one fixed schema (error: string). One commit, no
  cross-split mismatch.

Usage:
  python scripts/rebuild_results_hub.py --dry-run   # build + report, no writes
  python scripts/rebuild_results_hub.py             # wipe + push all splits
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RUN_TO_SPLIT = {
    "pilot": "pilot_v1",
    "main": "main_v1",
    "detector": "detector_v1",
    "baselines": "baselines_v1",
    "fp16": "fp16_v1",
    "scaling": "scaling_v1",
}


def to_records(results_list: list[dict], run_type: str, pv: str) -> list[dict]:
    """Fixed schema: scores and error always string so features match across splits."""
    out = []
    for r in results_list:
        scores = r.get("scores")
        scores_str = (
            "[]" if scores is None else (scores if isinstance(scores, str) else json.dumps(scores))
        )
        err = r.get("error")
        error_str = "" if err is None else (err if isinstance(err, str) else json.dumps(err))
        out.append(
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
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild results dataset on the Hub with a uniform schema.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--hub-repo", default=None, help="Default: harness.*.hub_repo from config.yaml.")
    ap.add_argument("--prompt-version", default="v1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root, pv = args.repo_root, args.prompt_version

    hub_repo = args.hub_repo
    if not hub_repo:
        import yaml

        cfg = root / "config.yaml"
        if not cfg.exists():
            raise SystemExit("config.yaml not found; pass --hub-repo.")
        data = yaml.safe_load(open(cfg))
        h = data.get("harness") or {}
        hub_repo = (
            (h.get("mlx") or {}).get("hub_repo")
            or (h.get("cuda") or {}).get("hub_repo")
            or (h.get("common") or {}).get("hub_repo")
            or ""
        ).strip()
    if not hub_repo:
        raise SystemExit("hub_repo not found; pass --hub-repo.")

    try:
        from datasets import Dataset, DatasetDict, Features, Value
    except ImportError:
        raise SystemExit("pip install datasets")

    feats = Features(
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

    splits = {}
    for run_type, split in RUN_TO_SPLIT.items():
        p = root / "results" / "past_runs" / run_type / "experiment_results.json"
        if not p.exists():
            print(f"  skip {split}: {p} missing")
            continue
        recs = to_records(json.load(open(p)), run_type, pv)
        splits[split] = Dataset.from_list(recs, features=feats)
        print(f"  {split}: {len(recs)} rows")

    if not splits:
        raise SystemExit("No local run files found under results/past_runs/.")

    if args.dry_run:
        print(f"Dry run: would rebuild {hub_repo} with splits {list(splits)} (no writes).")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    # Remove stale data + card so the null->string feature change is accepted.
    for path, is_folder in (("data", True), ("README.md", False)):
        try:
            if is_folder:
                api.delete_folder(
                    path, repo_id=hub_repo, repo_type="dataset",
                    commit_message="Clear old data for schema rebuild",
                )
            else:
                api.delete_file(
                    path, repo_id=hub_repo, repo_type="dataset",
                    commit_message="Clear old card for schema rebuild",
                )
            print(f"  cleared {path}")
        except Exception as e:
            print(f"  (nothing to clear at {path}: {type(e).__name__})")

    DatasetDict(splits).push_to_hub(
        hub_repo,
        private=True,
        commit_message="Rebuild: all splits, uniform schema (error: string).",
    )
    print(f"Pushed {len(splits)} splits to {hub_repo}: {list(splits)}")
    print("Next: upload your dataset card (README.md).")


if __name__ == "__main__":
    main()
