.PHONY: install install-spacy-model notebooks export open-export clean test

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

# Clean cached data
clean:
	rm -rf .cache/ results/
