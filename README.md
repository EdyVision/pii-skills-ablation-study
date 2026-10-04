# PII Agent Skills: Technical Report and Corrected Analysis

**Technical report:** *Evaluating Tool-Augmented PII Detection in Small Language Models*.

This project contains archived predictions and corrected analyses of zero-shot, documentation, tool, and Skill-based PII detection with Gemma 2 9B, Llama 3.1 8B, Mistral 7B and Qwen 2.5 7B. It describes the implemented pipelines; unequal context delivery and incomplete historical execution records prevent a clean causal interpretation.

## Results and limitations

All 32,000 primary records reproduce the headline means:

| Metric | Zero-shot | +Docs | +Tool | +Skills |
|---|---:|---:|---:|---:|
| Mean per-example type F1 | .631 | .602 | .527 | .511 |
| Mean per-example mixed span/type F1 | .325 | .301 | .455 | .448 |

The standalone detector baseline changes to **.502 type F1 / .453 mixed F1** after repairing missing gold annotations on 868 examples. Predictions are unchanged. The earlier large composed-system advantage and localization-parity claims are withdrawn. Diagnostic matching, volume denominators, precision-control uncertainty and inconsistent appendix scoring are corrected.

These results evaluate retained parsed outputs. Raw model responses were not archived, so historical parser losses cannot be reconstructed. The revised harness fixes known implementation defects; archived predictions were not regenerated with it. Tool-invocation and Skill-view flags are events, not proof of successful execution or equivalent document exposure. The 7B/14B adapters differ, and the precision control is limited to 300 examples.

## Data and scoring

The primary benchmark contains 2,000 examples: 1,132 AI4Privacy, 581 Gretel and 287 NVIDIA. It has 29 base categories and 27 after the DATE/DATE_TIME and ADDRESS/LOCATION bridges. Its sampling is specific to this compiled benchmark, not a population-representative guarantee.

The dehydrated benchmark contains source indices and labels, not input text or complete spans. Hydration must apply the original language/locale filtering before interpreting indices. Historical hydration lost NVIDIA offsets because serialized annotations were not JSON. The report metric uses span IoU >= .5 on 1,132 examples and type fallback on the remaining 868; it is not a pure span metric. Source labels agree with the primary gold, but six AI4Privacy examples have annotation text/offset disagreements; excluding them preserves the primary ranking.

- [Benchmark](https://huggingface.co/datasets/EdyVision/pii-skills-ablation)
- [Predictions and scores](https://huggingface.co/datasets/EdyVision/pii-skills-ablation-results)
- Report scoring contract: `lib/ablation_harness/scoring/saved_analysis.py`

The stored generic historical `scores` fields used older scoring conventions. Corrected exports explicitly identify the metric and version, include scoring gold, and distinguish type F1 from mixed span/type F1. Labels and offsets in historical predictions must not be rewritten to imply new inference. Detector entity text must be redacted before distribution.

## Offline reproduction

Use Python 3.11 or 3.12 and the project dependencies in `pyproject.toml`. The local `pii-codex` dependency points to a sibling checkout; supply that checkout or resolve the library explicitly for your environment.

Place archived local JSON records in `results/past_runs/{main,pilot,baselines,detector,fp16,scaling}/experiment_results.json`. The three analysis notebooks read these files and recompute scores with a shared contract. The detector loader joins validated primary gold by sample ID.

```sh
python -m pytest tests
python scripts/verify_report_notebooks.py
```

The checker runs the analysis notebooks with network calls and browser rendering disabled. It does not run inference or upload data. The corrected main, pilot and baseline/robustness notebooks contain refreshed outputs. `analysis-results/` and the notebook PDF figures contain derived summaries.

## Optional rerun preparation

CUDA support, run/seed provenance, offset-resolution controls and DGX configurations are retained in this repository. See [docs/DGX_RERUN.md](docs/DGX_RERUN.md). They are preparation for optional work, not completed experiments or evidence supporting the report. No new inference or multi-seed replication was performed during correction.

The supplied historical MLX configurations use one multi-turn worker. Hardware selection supports MLX on Apple Silicon, CUDA where available, and CPU otherwise. Backend changes and revised context delivery can change future predictions. Cooperative time checks do not forcibly cancel a blocking generation call.

## Historical files

Preparation and inference notebooks, prompts, configuration and older findings remain for provenance. Their older causal framing and results are superseded by the correction note and report. Changing `prompt_version` alone does not select a different prompt directory; set `prompts_dir` explicitly. Do not use old upload/rebuild scripts to republish unreviewed historical scores.

Source text is excluded from the dehydrated benchmark. Dataset users must follow each source's license and terms; redaction and public availability are not deployment-safety guarantees. Raw responses and source text should not be committed. Analysis supports an evaluation finding, not a claim that any pipeline safely removes all PII.

## Citation

```bibtex
@techreport{rosado2026context,
  title = {Evaluating Tool-Augmented PII Detection in Small Language Models},
  author = {Rosado, Eidan J.},
  year = {2026},
  note = {Technical report; not peer reviewed}
}
```

Cite the original source datasets as well. Publication metadata should be completed when the report is released. License: MIT for repository code; source dataset terms remain applicable.
