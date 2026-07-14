# PII Agent Skills Ablation Study

**Paper:** *Asymmetry, Not Capability: Evaluation Shapes Tool-Augmented PII Detection in Small Language Models*

Accompanying code, notebooks, and data for a controlled ablation study of whether structured knowledge documents (Agent Skills), documentation injection, and tool access improve small language model performance on PII detection — and how label alignment and evaluation granularity shape those conclusions.

## Research Question

**Does injecting Agent Skills, documentation, or tool access into SLM context improve PII detection accuracy compared to zero-shot prompting — and when do apparent gains or losses reflect measurement artifacts rather than true capability differences?**

## Experimental Design

**Ablation study** to isolate the effects of documentation and tool access:

| Condition | Documentation | Tool Access |
|-----------|---------------|-------------|
| Zero-shot | No | No |
| +Docs | Yes | No |
| +Tool | No | Yes |
| +Skills | Yes | Yes |

This design allows us to measure:
- **Main effect of documentation**: `+Docs` vs `Zero-shot`
- **Main effect of tool access**: `+Tool` vs `Zero-shot`
- **Interaction effect**: Whether documentation and tools together provide synergistic benefits beyond their individual contributions

**Baselines and controls.** Beyond the four-condition ablation, the study adds three reference baselines (a standalone PII-Codex detector with no model, few-shot prompting, and chain-of-thought prompting) and two robustness controls that test whether any effect is scale- or precision-bound: a larger model (Qwen 2.5 14B) and a full-precision (fp16/bf16) re-run of the four models.

## Models

Open-weight instruction-tuned models (~7-9B); same prompts and chat-template formatting for all (no model-specific scaffolding).

| Model | Parameters | Acceleration |
|-------|------------|--------------|
| Gemma 2 Instruct | 9B | MLX (Mac) / CPU |
| Llama 3.1 Instruct | 8B | MLX (Mac) / CPU |
| Mistral Instruct v0.3 | 7B | MLX (Mac) / CPU |
| Qwen 2.5 Instruct | 7B | MLX (Mac) / CPU |
| Qwen 2.5 Instruct *(14B scale control)* | 14B | MLX (Mac) / CPU |

The four 7–9B models are the primary set; Qwen 2.5 14B is a scale control, and a full-precision (fp16/bf16) re-run of the four primary models on a 300-sample subset is a precision control (see `config/rr_scaling.yaml`, `config/rr_fp16.yaml`).

Hardware is auto-detected: **MLX** on Darwin/arm64 (Mac), else **CPU**. CUDA is not yet supported (future research). Each model is unloaded before the next loads to avoid OOM when running all four locally.

**Inference batching:** Single-turn conditions (zero_shot, with_docs) are batched (progress bar over chunks). Multi-turn conditions (with_tools, with_skills) are per-sample (progress bar over samples). When both single-turn and multi-turn conditions are run, they execute in parallel (two threads, shared model; a lock serializes actual GPU calls so only one inference runs at a time). See `docs/FINDINGS.md` for details.

## Datasets

| Dataset | Domain | Size | Source |
|---------|--------|------|--------|
| AI4Privacy | Multi-domain | ~178k | HuggingFace |
| NVIDIA Nemotron-PII | Multi-domain | 100k | HuggingFace |
| Gretel PII masking | Multi-domain | 50k | HuggingFace |

## Sampling Strategy

Stratified sampling to ensure representativeness:

- **Strata**: PII type presence (which types appear) and text length bins (short/medium/long)
- **Pilot**: Configurable in `config.yaml` (`experiment.sampling.pilot_n`); e.g. n=200 for initial pilot
- **Target**: 500 samples per dataset if CI is acceptable (`target_n`); upper bound `max_n=2000` (main study)
- **Adaptive**: Increase N if variance is too high for reliable conclusions

All sampling uses a fixed seed for reproducibility.

## Data Preparation (Benchmark Compilation)

Notebook 1 builds a **dehydrated** benchmark (indices + labels only; text is hydrated at runtime in Notebook 2):

- **Language/locale filtration:** Each source is filtered to English-only before stratification and sampling (AI4Privacy: `language == "English"`; NVIDIA Nemotron-PII: `locale == "us"`; Gretel is already English-only). This is a **limitation**: the study does not support multiple languages or international PII types; results apply to English text only.
- **Label mapping**: Source PII labels are mapped to PII-Codex canonical types (`pii_codex_ground_truth`) so evaluation uses a single schema.
- **Exclusion**: Rows with **any** unmappable PII instance (source label not in the PII-Codex mapping) are excluded; only fully mappable rows are retained.
- **Fully labeled only**: Rows with no ground_truth are excluded before sampling and in top-up; a final safety filter drops any remaining empty-label rows. The benchmark never ships empty labels.
- **Minimum size**: The compiled benchmark has **at least 2,000** labeled records: the pool is oversampled per source (e.g. (2000 + 600) // 3 per source) so that after excluding empty and unmappable rows we still reach 2,000; we then cap at 2,000 or top up from larger extra draws (1,200 per source) until 2,000.
- **Uniqueness**: Each record is uniquely identified by `(source, original_index)`; the top-up step skips any duplicate and an assertion verifies no duplicates in the final dataset.

## Metrics

- **Two co-primary metrics**: entity-type match and span match (character IoU ≥ 0.5)
- **F1 Score**, **Precision**, and **Recall** under each metric
- A single **symmetric label-alignment** map applied identically to predictions and ground truth
- Per-condition and per-dataset breakdowns

## Configuration files

There are two kinds of config, in two places — same `HarnessConfig` underneath, different delivery:

- **`config.yaml` (repo root)** — the master study config (datasets, repos, sampling, models, and a nested `harness:` section). The notebooks read this as the single source of truth (`HarnessConfig.from_repo_config`).
- **`config/*.yaml` (the `config/` folder)** — flat, single-run configs passed to the CLI with `--config` (`HarnessConfig.from_yaml`). The harness also auto-discovers `config/<hardware>.yaml` here. The R&R run configs (`config/rr_baselines.yaml`, `config/rr_scaling.yaml`, `config/rr_fp16.yaml`) live here, alongside the label map (`config/pii_label_to_piicodex.json`).

Rule of thumb: edit `config.yaml` for the study defaults; add a flat `config/<name>.yaml` for a one-off run (e.g. a new model or precision) and point the CLI at it.

## Quick Start

The notebooks are standalone and install their own dependencies. Run directly in Colab:

| Notebook | Description |
|----------|-------------|
| `01_data_prep.ipynb` | Compile benchmark *(transparency only)* |
| `02_ablation_study.ipynb` | Run experiments (pilot or main) |
| `03_analyses_main.ipynb` | Score, visualize, upload (main study, n=2,000) |
| `03_analyses_pilot.ipynb` | Score, visualize, upload (pilot, n=200) |
| `04_analyses_r&r.ipynb` | Revise & resubmit analyses: symmetric alignment, dual (type/span) metrics, detector/few-shot/CoT baselines, and 14B + fp16 controls |

**Prerequisites:**
- Run `huggingface-cli login` or set `HF_TOKEN` environment variable

## Prompt Versioning

To test different prompt versions (e.g., after improving documentation):

1. **Set version in `config.yaml`**: Update `experiment.prompt_version` (e.g., `"v1"` → `"v2"`)
2. **Run notebook 1**: Uploads prompts to `prompts/v2/` on HuggingFace
3. **Run notebook 2**: Loads prompts from `prompts/v2/`, saves results to `split="v2"`
4. **Compare**: Load both `v1` and `v2` splits to compare performance

## HuggingFace Artifacts

| Repo | Contents | Created by |
|------|----------|------------|
| [`EdyVision/pii-skills-ablation`](https://huggingface.co/datasets/EdyVision/pii-skills-ablation) | Benchmark (sample indices + ground truth) and config (prompts, config.yaml, pii_label_to_piicodex.json under `config/`) | Notebook 1 |
| [`EdyVision/pii-skills-ablation-results`](https://huggingface.co/datasets/EdyVision/pii-skills-ablation-results) | Model predictions + scores (splits: `main_v1`, `pilot_v1`, `baselines_v1`, `detector_v1`, `fp16_v1`, `scaling_v1`) | Notebooks 2 & `04_analyses_r&r` |

## PII in Results

Benchmark samples and model outputs can contain real PII (e.g. IDs, passwords). When saving to `results/experiment_results.json`, we **redact** the `text` field in `predictions` and `ground_truth` (replace with `[REDACTED]`) so the saved file is safe to commit or publish. Type, start, and end are kept for scoring. `raw_response` is never written to disk. In-memory results during a run still contain full detail for debugging; do not export or share unredacted results from public repos.

## Data Licensing

Due to licensing restrictions, the benchmark dataset stores **sample indices only**, not raw text. This enables reproducibility while respecting data governance:

1. **Dehydrated dataset**: Contains `source`, `original_index`, and experiment results
2. **Hydration**: Notebook 02 fetches text from original datasets at runtime
3. **Requirements**: Users must have access to original datasets (accept terms where required)

This approach is standard practice for research involving restricted datasets.

## Citation

If you use this code, the benchmark, or the results, please cite the paper:

```bibtex
@article{rosado2026context,
  title   = {Asymmetry, Not Capability: Evaluation Shapes Tool-Augmented PII Detection in Small Language Models},
  author  = {Rosado, Eidan J.},
  year    = {2026}
}
```

Datasets:

- **Benchmark:** [EdyVision/pii-skills-ablation](https://huggingface.co/datasets/EdyVision/pii-skills-ablation)
- **Scored results:** [EdyVision/pii-skills-ablation-results](https://huggingface.co/datasets/EdyVision/pii-skills-ablation-results)

When using the benchmark, please also cite the source datasets (AI4Privacy PII-Masking-300K, NVIDIA Nemotron-PII, Gretel PII Masking); their BibTeX is in the [benchmark dataset card](https://huggingface.co/datasets/EdyVision/pii-skills-ablation). A `CITATION.cff` is included for GitHub's "Cite this repository" button.

## License

MIT
