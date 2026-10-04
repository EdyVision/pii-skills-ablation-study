# DGX rerun runbook

This runbook covers the corrected resubmission experiment. It does not change the model families used in the paper. The full run adds the existing Qwen 2.5 14B control; the manuscript must scope its scale conclusion to that within-family comparison.

## What the corrected harness records

- Native model offsets and exact, case-sensitive text-resolved offsets for every prediction.
- A privacy-safe resolution diagnostic, including ambiguity and resolution failure counts.
- Run ID, seed, decoding settings, prompt hashes, package versions, platform, and resolved Hugging Face model commit where available.
- Redacted local and Hub checkpoints; raw responses are represented only by SHA-256 hashes.

The equalized control never consults gold offsets or model-produced offsets. Repeated identical span text is assigned to occurrences from left to right. Missing text remains an unresolved prediction and therefore a false positive under span scoring.

## One-time setup and checks

Clone this repository beside `pii-codex`, because the current project configuration uses `../pii-codex` as its local source. Then run:

```bash
uv sync --extra dgx --extra dev
uv run pytest -q
uv run python -m ablation_harness.cli --help
```

Authenticate to Hugging Face before hydration or model loading. Models with gated weights require that the account has accepted their terms.

## Build the two immutable sample files

```bash
uv run python scripts/make_samples.py --out data/samples_main_corrected.json
uv run python scripts/make_samples.py --n 300 --seed 20260301 --out data/samples_seed300.json
```

Keep both files unchanged across runs. The 300-example builder preserves dataset-source proportions first and then the text-length/PII-signature mixture within each source.

Inspect the source counts and ground-truth validation flags before inference. Any annotation with `offset_text_matches=false` must be reported and handled by a prespecified sensitivity analysis rather than silently rewritten.

## Smoke gate

Before a full run, copy `config/dgx_primary.yaml` to a temporary config, set one model, one condition, and a temporary result directory, and use a small sample file. Verify:

- CUDA is selected and the intended dtype is shown.
- Parsed predictions contain text before persistence.
- Saved prediction text is `[REDACTED]`.
- `resolved_predictions`, `offset_resolution`, `scores`, `resolved_scores`, and `run_manifest.json` are present.
- A model error produces an explicit error row rather than dropping the sample.

Do not use smoke outputs in the paper.

## Corrected primary run

```bash
uv run python -m ablation_harness.cli \
  --config config/dgx_primary.yaml \
  --samples data/samples_main_corrected.json
```

This is the paper-grade greedy-decoding rerun across the four original models and Qwen 2.5 14B, all four ablation conditions, and the full benchmark.

## Seed-sensitivity runs

```bash
uv run python -m ablation_harness.cli --config config/dgx_seed42.yaml --samples data/samples_seed300.json
uv run python -m ablation_harness.cli --config config/dgx_seed43.yaml --samples data/samples_seed300.json
uv run python -m ablation_harness.cli --config config/dgx_seed44.yaml --samples data/samples_seed300.json
```

These runs use low-temperature sampling to make random-seed sensitivity identifiable. Analyze them as a separate robustness experiment; do not pool them with the greedy primary estimates.

## Required analysis gates

For each model, condition, and source, report entity-type and span precision, recall, and F1 for both native and text-resolved offsets. Use paired bootstrap confidence intervals over examples for condition contrasts and report the distribution across the three seed runs.

The central offset-provenance interpretation survives only if the detector-versus-zero-shot gap materially contracts under text-resolved zero-shot scoring without a comparable collapse in entity-type performance. If the gap does not contract, remove the provenance explanation and report the result as evidence against that hypothesis. If resolution failures or ambiguous repeated spans drive the result, report them and include a sensitivity analysis.

For scale, report Qwen 2.5 7B versus 14B as a single within-family control. Do not call it scale invariant. A broader scale claim requires at least one additional independently chosen larger-model comparison.
