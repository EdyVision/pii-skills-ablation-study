#!/usr/bin/env python3
"""
Hydrate the compiled benchmark into a samples JSON the harness CLI can consume.

Each sample dict has: id, text, source, ground_truth (JSON string with spans), and
pii_codex_ground_truth (if present in the benchmark). This mirrors Notebook 2's hydration
(`hydrate_sample`) so a terminal run produces the SAME records as the in-notebook run.

Default: the full benchmark (the same 2,000 used for the main run) -> a directly comparable
scaling run. Use --n for a stratified subsample (e.g. the fp16 full-precision check), which
reuses Notebook 2's stratified_subsample with the same seed.

Usage:
  python scripts/make_samples.py --out data/samples_main.json
  python scripts/make_samples.py --n 300 --out data/samples_sub300.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset


def hydrate(sample: dict, ai4, nvidia, gretel) -> dict:
    """Fetch text + ground_truth from the original source datasets (mirror of Notebook 2)."""
    source = sample["source"]
    idx = sample["original_index"]
    text = None
    gt: list[dict] = []

    if source == "ai4privacy":
        row = ai4[idx]
        text = row["source_text"]
        for item in row.get("privacy_mask") or []:
            if isinstance(item, dict):
                gt.append(
                    {
                        "type": item.get("label", "UNK"),
                        "text": item.get("value", ""),
                        "start": item.get("start", 0),
                        "end": item.get("end", 0),
                    }
                )
    elif source == "nvidia_nemotron_pii":
        row = nvidia[idx]
        text = row.get("text", "")
        spans = row.get("spans")
        if isinstance(spans, str):
            try:
                spans = json.loads(spans)
            except (json.JSONDecodeError, TypeError):
                spans = []
        for s in spans if isinstance(spans, list) else []:
            if isinstance(s, dict):
                gt.append(
                    {
                        "type": s.get("label", "UNK"),
                        "text": s.get("text", ""),
                        "start": s.get("start", 0),
                        "end": s.get("end", 0),
                    }
                )
    elif source == "gretel_pii_masking":
        row = gretel[idx]
        text = row.get("text", "")
        for e in row.get("entities") or []:
            if isinstance(e, dict):
                entity_text = e.get("entity", "")
                for t in e.get("types") or []:
                    gt.append({"type": t, "text": entity_text, "start": 0, "end": 0})

    sample["text"] = text
    sample["ground_truth"] = json.dumps(gt)
    return sample


def _len_bin(text: str) -> str:
    n = len(text or "")
    return "short" if n < 500 else "medium" if n < 2000 else "long"


def _pii_sig(gt_val) -> str:
    try:
        gt = json.loads(gt_val) if isinstance(gt_val, str) else (gt_val or [])
    except (json.JSONDecodeError, TypeError):
        gt = []
    types = sorted({i.get("type", "UNK") for i in gt if isinstance(i, dict)})
    return "|".join(types[:3]) if types else "none"


def stratified_subsample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by (source, length bin, pii signature) — same logic as Notebook 2."""
    N = len(rows)
    if n >= N:
        return rows
    strata: dict = {}
    for i, row in enumerate(rows):
        key = (row.get("source", "unknown"), _len_bin(row.get("text") or ""), _pii_sig(row.get("ground_truth")))
        strata.setdefault(key, []).append(i)
    random.seed(seed)
    selected: list[int] = []
    for idxs in strata.values():
        share = max(1, min(int(n * len(idxs) / N), len(idxs)))
        random.shuffle(idxs)
        selected.extend(idxs[:share])
    if len(selected) > n:
        random.shuffle(selected)
        selected = selected[:n]
    elif len(selected) < n:
        remaining = [i for i in range(N) if i not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    random.shuffle(selected)
    return [rows[i] for i in selected]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Hydrate benchmark -> samples JSON for the harness CLI.")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. data/samples_main.json)")
    ap.add_argument("--n", type=int, default=None, help="Stratified subsample size (default: full benchmark)")
    ap.add_argument("--seed", type=int, default=None, help="Subsample seed (default: experiment.seed from config.yaml)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(repo_root / "config.yaml"))
    benchmark_repo = cfg["compiled_dataset"]["repo_id"]
    seed = args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42)

    print(f"Loading benchmark: {benchmark_repo} (split=test)")
    bench = load_dataset(benchmark_repo, split="test")
    print(f"  {len(bench)} benchmark records")

    print("Loading source datasets for hydration (one-time download) ...")
    ai4 = load_dataset("ai4privacy/pii-masking-300k", split="train")
    nvidia = load_dataset("nvidia/Nemotron-PII", split="train")
    gretel = load_dataset("gretelai/gretel-pii-masking-en-v1", split="train")

    rows: list[dict] = []
    for r in bench:
        s = dict(r)
        s = hydrate(s, ai4, nvidia, gretel)
        if s.get("text") is None:
            continue
        s["id"] = s.get("sample_id", s.get("id", str(len(rows))))
        rows.append(s)
    print(f"Hydrated {len(rows)} / {len(bench)} samples")

    if args.n is not None:
        rows = stratified_subsample(rows, args.n, seed)
        print(f"Stratified subsample -> {len(rows)} samples (seed={seed})")

    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"Wrote {len(rows)} samples to {out}")


if __name__ == "__main__":
    main()
