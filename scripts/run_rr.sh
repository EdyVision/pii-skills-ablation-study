#!/usr/bin/env bash
#
# R&R run launcher — headless, isolated, additive. Drives any rr config.
#
# What it does:
#   1. Backs up the existing runs (results/ is gitignored, so this is the safety net).
#   2. Launches ONLY the models in CONFIG across all conditions, writing to that config's
#      results_dir and pushing NOWHERE (push_to_hub: false in the config).
#   3. Wraps the run in `caffeinate` (macOS) so the machine won't idle-sleep mid-run, and
#      detaches via `nohup` so you can close the terminal.
#
# It does NOT hydrate samples — build those first (make rr-samples / make rr-samples-fp16),
# so the long download finishes while you watch and this step detaches instantly.
#
# Usage (or use the Makefile targets, which set these for you):
#   CONFIG=config/rr_scaling.yaml SAMPLES=data/samples_main.json   bash scripts/run_rr.sh
#   CONFIG=config/rr_fp16.yaml    SAMPLES=data/samples_sub300.json bash scripts/run_rr.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
CONFIG="${CONFIG:-config/rr_scaling.yaml}"
SAMPLES="${SAMPLES:-data/samples_main.json}"

# Derive the output dir + a label from the config (so logs/backups are named per run).
RESULTS_DIR="$(grep '^results_dir:' "$CONFIG" | awk '{print $2}')"
RUN_TYPE="$(grep '^run_type:' "$CONFIG" | awk '{print $2}')"
LOG="${RESULTS_DIR}/run_${RUN_TYPE}_${TS}.log"
mkdir -p "$RESULTS_DIR" data

# Guardrail echo: confirm what will run and that it won't push.
echo "Config:      $CONFIG"
echo "Models:      $(grep -A6 '^models:' "$CONFIG" | grep '^  - ' | tr -d ' -' | paste -sd, -)"
echo "run_type:    $RUN_TYPE   -> Hub split would be ${RUN_TYPE}_v1 (not pushed)"
echo "push_to_hub: $(grep '^push_to_hub:' "$CONFIG" | awk '{print $2}')"
echo "results_dir: $RESULTS_DIR"
echo "samples:     $SAMPLES"
echo

if [ ! -f "$SAMPLES" ]; then
  echo "ERROR: samples file not found: $SAMPLES"
  echo "Build it first:  make rr-samples   (full 2,000)   or   make rr-samples-fp16   (subsample)"
  exit 1
fi

# Backup existing runs (timestamped).
echo "Backing up existing runs -> results/past_runs/_backup_${TS}.tar.gz"
tar czf "results/past_runs/_backup_${TS}.tar.gz" -C results past_runs/main past_runs/pilot 2>/dev/null || true

# Stay-awake wrapper (macOS only) + project env runner.
WAKE=""
command -v caffeinate >/dev/null 2>&1 && WAKE="caffeinate -i"
RUNNER="python -m ablation_harness"
command -v uv >/dev/null 2>&1 && RUNNER="uv run python -m ablation_harness"

echo "Launching ${RUN_TYPE} run (detached, stay-awake). Log: $LOG"
PYTHONPATH=lib nohup $WAKE $RUNNER \
  --config "$CONFIG" \
  --samples "$SAMPLES" \
  --repo "$REPO" \
  > "$LOG" 2>&1 &

PID=$!
echo "Started (PID $PID)."
echo "Watch:   tail -f $LOG       (or: make rr-log)"
echo "Output:  ${RESULTS_DIR}/experiment_results.json"
echo "Analyze: notebooks/04_analyses_r&r.ipynb -> load_run(\"${RUN_TYPE}\")"
