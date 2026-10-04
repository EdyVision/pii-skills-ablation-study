.PHONY: install install-spacy-model notebooks export open-export clean test \
        rr-samples rr-samples-fp16 rr-scaling rr-fp16 rr-baselines rr-log rr-upload rr-upload-dry rr-rebuild-hub rr-rebuild-hub-dry

# Install all dependencies (MLX, pii-codex[detections]>=0.6.1, etc.) via uv, then spaCy English model
install:
	uv sync
	$(MAKE) install-spacy-model

# Install spaCy English model (required for pii-codex detections; not on PyPI as a direct dep)
install-spacy-model:
	uv pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"

# Run jupyter via uv
notebooks:
	uv run jupyter notebook notebooks/

# Export notebooks to HTML (output only, no code)
export:
	jupyter nbconvert --to html --no-input notebooks/01_data_prep.ipynb --output-dir=results/
	jupyter nbconvert --to html --no-input notebooks/02_ablation_study.ipynb --output-dir=results/

# Export notebooks to HTML and open them in the default browser
open-export: export
	open results/01_data_prep.html results/02_ablation_study.html

# Run tests (sync with dev deps then pytest via uv so project deps e.g. rich are available)
test:
	uv sync --all-extras
	uv run pytest tests/ -v

# ---- R&R (revise & resubmit) runs ----------------------------------------------------
# 1) Build the samples file (hydrates benchmark text). Do this first; it downloads sources.
rr-samples:           ## full 2,000 samples for the 14B scaling run
	uv run python scripts/make_samples.py --out data/samples_main.json

rr-samples-fp16:      ## stratified 300-sample subset for the full-precision check
	uv run python scripts/make_samples.py --n 300 --out data/samples_sub300.json

# 2) Launch a run (detached, stay-awake, writes to results/past_runs/<run_type>/, no Hub push).
rr-scaling: ## 14B-4bit across all four conditions (reviewer Q4)
	CONFIG=config/rr_scaling.yaml SAMPLES=data/samples_main.json bash scripts/run_rr.sh

rr-fp16:    ## four models at full precision on the subsample (reviewer Q5)
	CONFIG=config/rr_fp16.yaml SAMPLES=data/samples_sub300.json bash scripts/run_rr.sh

rr-baselines: ## few-shot + chain-of-thought baselines, four models, full 2,000 (reviewer 2 #6)
	CONFIG=config/rr_baselines.yaml SAMPLES=data/samples_main.json bash scripts/run_rr.sh

# 4) Push the R&R runs to the results dataset on the Hub (main/pilot are already there).
#    Run `make rr-upload-dry` first to preview split names and row counts.
RR_RUNS ?= detector baselines fp16 scaling
PV ?= v1

rr-upload-dry: ## preview which R&R splits would be pushed (no writes)
	@for rt in $(RR_RUNS); do \
	  uv run python scripts/upload_results_to_hub.py --run-type $$rt --prompt-version $(PV) --dry-run; \
	done

rr-upload: ## push detector + baselines + fp16 + scaling splits to the results dataset
	@for rt in $(RR_RUNS); do \
	  echo "==> $$rt ($(PV))"; \
	  uv run python scripts/upload_results_to_hub.py --run-type $$rt --prompt-version $(PV) || exit 1; \
	done

# 4b) One-shot rebuild of the whole results dataset with a uniform schema (error: string).
#     Use this when the Hub has old null-error splits that block incremental rr-upload.
rr-rebuild-hub-dry: ## preview the full-rebuild push (no writes)
	uv run python scripts/rebuild_results_hub.py --dry-run

rr-rebuild-hub: ## wipe + re-push ALL splits in one DatasetDict (then re-upload README)
	uv run python scripts/rebuild_results_hub.py

# 3) Tail the most recent run log.
rr-log:
	@tail -f "$$(ls -t results/past_runs/*/run_*.log | head -1)"

# Clean cached data. NOTE: deliberately does NOT touch results/past_runs (your run data).
clean:
	rm -rf .cache/
	rm -f results/*.html
