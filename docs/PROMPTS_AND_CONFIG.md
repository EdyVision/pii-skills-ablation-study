# Prompt and Config Hosting Strategy

Benchmark and config live in one repo (`EdyVision/pii-skills-ablation`). Config files (prompts, config.yaml, label mapping) are under a `config/` prefix so they don’t collide with the dataset. This allows prompt/config updates without re-uploading the full dataset and keeps everything portable via HuggingFace Hub.

| Repo | Contents |
|------|----------|
| `EdyVision/pii-skills-ablation` | Benchmark (dataset split) + config under `config/` |
| `EdyVision/pii-skills-ablation-results` | Model predictions + scores |

### Structure

```
EdyVision/pii-skills-ablation/
  test/                    # Dataset split (benchmark)
  config/
    config.yaml
    pii_label_to_piicodex.json
    prompts/
      v1/
        zero_shot.txt
        with_docs.txt
        with_skills.txt
```

### Usage

**Notebook 01 (Data Prep):** Uploads benchmark then prompts and config to the same repo (config under `config/`). Notebook included for transparency and reproducibility.

**Loading config from HuggingFace:**

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="EdyVision/pii-skills-ablation",
    filename="config/prompts/v1/zero_shot.txt",
    repo_type="dataset"
)
```