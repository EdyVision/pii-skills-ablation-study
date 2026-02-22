"""Experiment runner — run_experiments, _run_conditions_batch, run_single_experiment."""

from __future__ import annotations

import dataclasses
import gc
import json
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ablation_harness.config import HarnessConfig
from ablation_harness.loops.single_turn import run_single_turn
from ablation_harness.model import ModelInference
from ablation_harness.parsing.json_response import parse_json_response
from ablation_harness.scoring.metrics import compute_metrics, ground_truth_for_scoring
from ablation_harness.scoring.normalize import load_label_map
from ablation_harness.tools.registry import ToolRegistry, PiiCodexTool
from skill_agent import run_skill_agent

try:
    from pii_codex.services.analysis_service import PIIAnalysisService
except ImportError:
    PIIAnalysisService = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore[misc, assignment]


class _LockedModel:
    """Wraps ModelInference so only one thread calls generate/generate_batch at a time."""

    def __init__(self, model, lock):
        self._model = model
        self._lock = lock

    def generate(self, prompt, max_tokens):
        with self._lock:
            return self._model.generate(prompt, max_tokens)

    def generate_batch(self, prompts, max_tokens):
        with self._lock:
            return self._model.generate_batch(prompts, max_tokens)

    def __getattr__(self, name):
        return getattr(self._model, name)


def load_prompts(config: HarnessConfig) -> dict[str, str]:
    """Load prompt templates from the prompts directory."""
    prompts = {}
    for condition in config.conditions:
        prompt_path = config.prompts_dir / f"{condition}.txt"
        if prompt_path.exists():
            prompts[condition] = prompt_path.read_text()
        else:
            raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompts


def _condition_to_flags(condition: str) -> tuple[bool, bool, bool]:
    """Map condition name to (with_skills, with_docs, with_tools). None True => zero_shot."""
    if condition == "zero_shot":
        return False, False, False
    if condition == "with_docs":
        return False, True, False
    if condition == "with_tools":
        return False, False, True
    if condition == "with_skills":
        return True, True, True
    return False, False, False


def run_single_experiment(
    model,
    sample: dict,
    condition: str,
    config: HarnessConfig,
    prompts: dict[str, str],
    tool_registry: ToolRegistry,
    *,
    model_name: str | None = None,
) -> dict:
    """Run experiment on a single sample."""
    prompt = prompts[condition].format(text=sample["text"])

    try:
        custom_runner = getattr(config, "with_skills_runner", None)
        if not callable(custom_runner) and condition in ("with_skills", "with_tools"):
            raise RuntimeError(
                "with_skills and with_tools require config.with_skills_runner to be set "
                "(e.g. runner.config.with_skills_runner = run_skill_agent)."
            )
        if callable(custom_runner):
            with_skills, with_docs, with_tools = _condition_to_flags(condition)
            return custom_runner(
                model,
                sample,
                prompt,
                config,
                tool_registry,
                with_skills=with_skills,
                with_docs=with_docs,
                with_tools=with_tools,
                model_name=model_name,
            )
        # zero_shot and with_docs only (with_skills/with_tools require with_skills_runner)
        return run_single_turn(model, prompt, config.max_tokens)
    except Exception as e:
        return {
            "predictions": [],
            "raw_response": None,
            "error": str(e),
        }


def _run_conditions_batch(
    model,
    model_name: str,
    samples: list,
    conditions: list[str],
    config: HarnessConfig,
    prompts: dict[str, str],
    tool_registry: ToolRegistry,
    batch_size: int | None = None,
    shared_progress=None,
    shared_task_id=None,
) -> list[dict]:
    """Run a list of conditions sequentially; returns flat list of result dicts.

    Progress is driven by the caller via shared_progress / shared_task_id
    (one bar per model). This function only advances that bar.
    """
    bs = batch_size if batch_size is not None else config.batch_size
    progress = shared_progress
    task = shared_task_id
    out: list[dict] = []

    for condition in conditions:
        condition_results: list[dict] = []

        # zero_shot and with_docs always use batched path (batch_size); with_tools/with_skills use custom runner.
        if condition in ("zero_shot", "with_docs"):
            chunks = [
                [samples[i + j] for j in range(min(bs, len(samples) - i))]
                for i in range(0, len(samples), bs)
            ]

            for chunk in chunks:
                chunk_prompts = [
                    prompts[condition].format(text=s["text"]) for s in chunk
                ]
                t0 = time.perf_counter()
                try:
                    responses = model.generate_batch(chunk_prompts, config.max_tokens)
                except Exception:
                    # Sequential fallback
                    for sample, prompt in zip(chunk, chunk_prompts):
                        t0s = time.perf_counter()
                        turns = 1
                        try:
                            response = model.generate(prompt, config.max_tokens)
                            predictions = parse_json_response(response)
                            result = {
                                "predictions": predictions,
                                "raw_response": response,
                                "tool_executed": False,
                                "skill_viewed": False,
                                "conversation_turns": turns,
                                "error": None,
                            }
                        except Exception as e2:
                            result = {
                                "predictions": [],
                                "raw_response": None,
                                "tool_executed": False,
                                "skill_viewed": False,
                                "conversation_turns": turns,
                                "error": str(e2),
                            }
                        result["elapsed_seconds"] = time.perf_counter() - t0s
                        result["sample_id"] = sample["id"]
                        result["model"] = model_name
                        result["condition"] = condition
                        result["source"] = sample["source"]
                        result["ground_truth"] = ground_truth_for_scoring(sample)
                        condition_results.append(result)
                        if progress is not None:
                            progress.update(1)
                    continue

                batch_elapsed = time.perf_counter() - t0
                per_sample_sec = batch_elapsed / len(chunk)
                for sample, response in zip(chunk, responses):
                    turns = 1
                    if response is None:
                        result = {
                            "predictions": [],
                            "raw_response": None,
                            "tool_executed": False,
                            "skill_viewed": False,
                            "conversation_turns": turns,
                            "error": "Batch returned None response",
                        }
                    else:
                        predictions = parse_json_response(response)
                        result = {
                            "predictions": predictions,
                            "raw_response": response,
                            "tool_executed": False,
                            "skill_viewed": False,
                            "conversation_turns": turns,
                            "error": None,
                        }
                    result["elapsed_seconds"] = per_sample_sec
                    result["sample_id"] = sample["id"]
                    result["model"] = model_name
                    result["condition"] = condition
                    result["source"] = sample["source"]
                    result["ground_truth"] = ground_truth_for_scoring(sample)
                    condition_results.append(result)
                if progress is not None:
                    progress.update(len(chunk))
        else:
            # Multi-turn
            def worker(sample):
                t0 = time.perf_counter()
                result = run_single_experiment(
                    model,
                    sample,
                    condition,
                    config,
                    prompts,
                    tool_registry,
                    model_name=model_name,
                )
                result["elapsed_seconds"] = time.perf_counter() - t0
                result["sample_id"] = sample["id"]
                result["model"] = model_name
                result["condition"] = condition
                result["source"] = sample["source"]
                result["ground_truth"] = ground_truth_for_scoring(sample)
                if progress is not None:
                    progress.update(1)
                return result

            with ThreadPoolExecutor(max_workers=config.multi_turn_workers) as executor:
                futures = [executor.submit(worker, s) for s in samples]
                condition_results = [f.result() for f in futures]

        out.extend(condition_results)

    gc.collect()
    if config.hardware == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def _redact_pii_in_list(items):
    if not items or not isinstance(items, list):
        return items
    return [
        ({**d, "text": "[REDACTED]"} if isinstance(d, dict) and "text" in d else d)
        for d in items
    ]


def save_results_to_disk(config: HarnessConfig, results_list: list[dict]) -> Path:
    """Write results to config.results_dir / experiment_results.json (PII redacted). Returns path."""
    results_path = config.results_dir / "experiment_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    slim = []
    for r in results_list:
        row = {k: v for k, v in r.items() if k != "raw_response"}
        for key in ("predictions", "ground_truth"):
            if key in row and row[key] is not None:
                row[key] = _redact_pii_in_list(row[key])
        slim.append(row)
    with open(results_path, "w") as f:
        json.dump(slim, f, indent=2)
    return results_path


def push_results_to_hub(config: HarnessConfig, results_list: list[dict]) -> None:
    """Push results to Hugging Face hub (same format as old notebook for notebook 3). No-op if push_to_hub false or Dataset unavailable."""
    if not getattr(config, "push_to_hub", False) or not (
        getattr(config, "hub_repo", "") or not config.hub_repo.strip()
    ):
        return
    if Dataset is None:
        print("  Push to Hub skipped: datasets package required (pip install datasets)")
        return
    run_type = getattr(config, "run_type", "pilot")
    pv = getattr(config, "prompt_version", "v2")
    split_name = f"{run_type}_{pv}"
    try:
        records = [
            {
                "sample_id": r.get("sample_id"),
                "source": r.get("source"),
                "model": r.get("model"),
                "condition": r.get("condition"),
                "prompt_version": pv,
                "run_type": run_type,
                "predictions": json.dumps(r.get("predictions") or []),
                "scores": (
                    json.dumps(r.get("scores")) if r.get("scores") is not None else ""
                ),
                "error": r.get("error"),
                "tool_executed": bool(r.get("tool_executed", False)),
                "skill_viewed": bool(r.get("skill_viewed", False)),
                "elapsed_seconds": r.get("elapsed_seconds"),
                "conversation_turns": r.get("conversation_turns"),
            }
            for r in results_list
        ]
        ds = Dataset.from_list(records)
        ds.push_to_hub(
            config.hub_repo.strip(),
            split=split_name,
            private=True,
            commit_message=f"Checkpoint: {len(records)} results ({run_type} {pv})",
        )
        print(
            f"  Pushed to Hub: {config.hub_repo} (split={split_name}, n={len(records)})"
        )
    except Exception as e:
        print(f"  Push to Hub skipped: {e}")


def _load_existing_results(config: HarnessConfig) -> list[dict]:
    """Load existing results for merge: local file first; if missing, try Hub. Returns [] if nothing found (local and Hub empty or missing)."""
    results_path = config.results_dir / "experiment_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)

    hub_repo = getattr(config, "hub_repo", "") or ""
    if not getattr(config, "push_to_hub", False) or not hub_repo.strip():
        return []

    run_type = getattr(config, "run_type", "pilot")
    pv = getattr(config, "prompt_version", "v1")
    split_name = f"{run_type}_{pv}"

    try:
        from huggingface_hub import list_repo_files, repo_exists

        if not repo_exists(hub_repo.strip(), repo_type="dataset"):
            return []
        files = list_repo_files(hub_repo.strip(), repo_type="dataset")
        # Only load if this split exists on Hub (e.g. data/main_v1-*.parquet); else return [].
        split_exists = any(
            f.startswith(f"data/{split_name}-") or f == f"data/{split_name}.parquet"
            for f in files
        )
        if not split_exists:
            print(
                f"  Split {split_name} not on Hub yet; no existing results to merge (first push will create it).",
                flush=True,
            )
            return []
    except Exception:
        return []

    try:
        from datasets import load_dataset as _load_ds

        print(
            f"  Loading existing results from Hub (split={split_name})...", flush=True
        )
        ds = _load_ds(
            hub_repo.strip(), split=split_name, download_mode="force_redownload"
        )
    except Exception:
        return []

    if ds is None or len(ds) == 0:
        return []

    out = []
    for i in range(len(ds)):
        row = ds[i]
        preds = row.get("predictions")
        scores = row.get("scores")
        if isinstance(preds, str):
            try:
                preds = json.loads(preds)
            except (TypeError, json.JSONDecodeError):
                preds = []
        if isinstance(scores, str) and scores:
            try:
                scores = json.loads(scores)
            except (TypeError, json.JSONDecodeError):
                scores = None
        elif isinstance(scores, str) and not scores:
            scores = None
        out.append(
            {
                "sample_id": row.get("sample_id"),
                "source": row.get("source"),
                "model": row.get("model"),
                "condition": row.get("condition"),
                "predictions": preds,
                "scores": scores,
                "error": row.get("error"),
                "ground_truth": None,
            }
        )
    return out


def load_results(path: Path | str) -> list[dict]:
    """Load results from experiment_results.json (local file). For Hub, use datasets.load_dataset then convert rows to this format."""
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def score_results(
    results: list[dict],
    label_map_path: Path | str,
) -> list[dict]:
    """Add 'scores' to each result using label_map. Mutates and returns *results*. Use when analyzing without re-running (e.g. notebook 3 pulls from Hub)."""
    label_map_path = Path(label_map_path)
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found: {label_map_path}")
    label_map = load_label_map(label_map_path)
    for r in results:
        preds = r.get("predictions")
        gt = r.get("ground_truth", [])
        if isinstance(preds, str):
            try:
                preds = json.loads(preds)
            except (TypeError, json.JSONDecodeError):
                preds = []
        if isinstance(gt, str):
            try:
                gt = json.loads(gt)
            except (TypeError, json.JSONDecodeError):
                gt = []
        r["scores"] = compute_metrics(preds or [], gt, label_map)
    return results


def run_experiments(
    config: HarnessConfig,
    samples: list[dict],
    prompts: dict[str, str],
    tool_registry: ToolRegistry,
    *,
    merge_with_saved: bool = True,
) -> list[dict]:
    """Run experiments for all models and conditions specified in *config*.

    If merge_with_saved is True (default), load existing results from
    config.results_dir / experiment_results.json, drop rows for the (model, condition)
    pairs being run, run those, merge with existing, and save/push. This avoids
    overwriting previous runs when re-running a subset of conditions.
    """
    # Future research to include CUDA; only MLX (and optionally CPU) have been validated.
    if config.hardware == "cuda":
        raise NotImplementedError(
            "CUDA path is not yet validated. Use hardware='mlx' or 'cpu'. Future research to include CUDA."
        )
    _rt = getattr(config, "run_type", "pilot")
    _pv = getattr(config, "prompt_version", "v1")
    print(
        f"run_experiments: run_type={_rt!r} (Hub split will be {_rt}_{_pv})", flush=True
    )
    run_conditions = config.conditions
    re_run_pairs = {(m, c) for m in config.models for c in run_conditions}
    existing: list[dict] = []
    if merge_with_saved:
        raw = _load_existing_results(config)
        existing = [
            r for r in raw if (r.get("model"), r.get("condition")) not in re_run_pairs
        ]
        if existing:
            print(
                f"Merging with {len(existing)} existing results "
                f"(re-running {run_conditions} for {list(config.models)})."
            )
    all_results: list[dict] = list(existing)

    def _save_and_push(results_list):
        save_results_to_disk(config, results_list)
        push_results_to_hub(config, results_list)

    run_conditions = config.conditions

    for model_name in config.models:
        print(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")
        model_config = config.model_configs.get(model_name, {})
        model = ModelInference(model_name, config.hardware, model_config)
        model.load()

        # Single progress bar per model — tracks samples across all conditions.
        total_samples = len(run_conditions) * len(samples)
        inference_lock = threading.Lock()
        model_for_inference = _LockedModel(model, inference_lock)
        bs_override = None
        if config.hardware != "mlx":
            bs_override = config.batch_size_parallel

        pbar = tqdm(total=total_samples, desc=model_name, unit="sample")
        for condition in run_conditions:
            bs_arg = (
                bs_override
                if bs_override and condition in ("zero_shot", "with_docs")
                else None
            )
            condition_results = _run_conditions_batch(
                model_for_inference,
                model_name,
                samples,
                [condition],
                config,
                prompts,
                tool_registry,
                bs_arg,
                shared_progress=pbar,
                shared_task_id=None,
            )
            all_results.extend(condition_results)
            _save_and_push(all_results)
            errors = sum(1 for r in condition_results if r["error"])
            completed = len(condition_results) - errors
            total_sec = sum(r.get("elapsed_seconds", 0) for r in condition_results)
            suffix = (
                " — saved & pushed"
                if config.push_to_hub and config.hub_repo
                else " — saved"
            )
            if condition in ("with_tools", "with_skills"):
                tool_calls = sum(
                    1 for r in condition_results if r.get("tool_executed", False)
                )
                pbar.write(
                    f"  {condition}: {completed} done, {errors} errors, "
                    f"tool_calls {tool_calls}/{len(condition_results)}, "
                    f"{total_sec:.1f}s{suffix}"
                )
            else:
                pbar.write(
                    f"  {condition}: {completed} done, {errors} errors, "
                    f"{total_sec:.1f}s{suffix}"
                )
        pbar.close()
        _save_and_push(all_results)
        print(
            f"  Checkpoint: saved and pushed ({len(all_results)} results so far)",
            flush=True,
        )

        model.unload()
        gc.collect()
        del model

    _save_and_push(all_results)
    results_path = config.results_dir / "experiment_results.json"
    hub_note = " and pushed to Hub" if config.push_to_hub and config.hub_repo else ""
    print(
        f"\nResults saved to {results_path} ({len(all_results)} total, PII text redacted){hub_note}"
    )

    # Summary table
    agg = defaultdict(
        lambda: {"n": 0, "errors": 0, "tool_calls": 0, "elapsed_seconds": 0.0}
    )
    for r in all_results:
        key = (r["model"], r["condition"])
        agg[key]["n"] += 1
        if r.get("error"):
            agg[key]["errors"] += 1
        if r.get("tool_executed", False):
            agg[key]["tool_calls"] += 1
        agg[key]["elapsed_seconds"] += r.get("elapsed_seconds", 0)
    rows = [
        {
            "model": m,
            "condition": c,
            "n": v["n"],
            "errors": v["errors"],
            "tool_calls": v["tool_calls"],
            "total_seconds": round(v["elapsed_seconds"], 1),
            "mean_seconds": round(v["elapsed_seconds"] / v["n"], 1) if v["n"] else 0,
        }
        for (m, c), v in sorted(agg.items())
    ]
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    return all_results


def _resolve_config_paths(config: HarnessConfig, repo_root: Path) -> None:
    """Resolve relative paths in config against repo_root."""
    for key in ("prompts_dir", "skills_dir", "results_dir", "label_map_path"):
        val = getattr(config, key, None)
        if val is not None and not Path(val).is_absolute():
            setattr(config, key, repo_root / val)


class AblationRunner:
    """Callable runner for notebook/Colab: one object to load config, prompts, registry, run experiments, and score."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        config_path: Path | str | None = None,
        repo_root: Path | str | None = None,
        register_pii_tool: bool = True,
    ):
        repo_root = Path(repo_root) if repo_root is not None else Path.cwd()
        if config is not None:
            self.config = config
            self._repo_root = repo_root
            _resolve_config_paths(self.config, self._repo_root)
        elif config_path is not None:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                config_path = repo_root / config_path
            self.config = HarnessConfig.from_yaml(config_path)
            self._repo_root = repo_root
            _resolve_config_paths(self.config, self._repo_root)
        else:
            self._repo_root = repo_root
            self.config = HarnessConfig.from_detected(repo_root / "config")
            _resolve_config_paths(self.config, self._repo_root)
        # Default to skill agent (required dependency)
        if getattr(self.config, "with_skills_runner", None) is None:
            self.config.with_skills_runner = run_skill_agent
        self.register_pii_tool = register_pii_tool

    def run(
        self,
        samples: list[dict],
        *,
        score: bool = True,
        conditions: list[str] | None = None,
        merge_with_saved: bool = True,
        run_type: str | None = None,
    ) -> list[dict]:
        """Run experiments on *samples* and optionally score; returns list of result dicts.

        By default runs all conditions from config (zero_shot, with_docs, with_tools, with_skills).
        Pass conditions= to run only specific conditions, e.g. conditions=[\"with_skills\"].
        Pass run_type=\"main\" or \"pilot\" so Hub split and merge use that run (e.g. main_v1 vs pilot_v1).
        If merge_with_saved is True (default), load existing results from disk, drop only the
        (model, condition) pairs being run, run those, merge, and save/push so previous runs
        are not overwritten.
        """
        if run_type is not None:
            self.config.run_type = run_type
        run_config = (
            dataclasses.replace(self.config, conditions=conditions)
            if conditions is not None
            else self.config
        )
        if run_type is not None:
            run_config.run_type = run_type
        prompts = load_prompts(run_config)
        registry = ToolRegistry()
        if self.register_pii_tool and PIIAnalysisService is not None:
            registry.register(PiiCodexTool(PIIAnalysisService()))
        results = run_experiments(
            run_config,
            samples,
            prompts,
            registry,
            merge_with_saved=merge_with_saved,
        )
        if score and self.config.label_map_path and self.config.label_map_path.exists():
            label_map = load_label_map(self.config.label_map_path)
            for r in results:
                r["scores"] = compute_metrics(
                    r["predictions"], r.get("ground_truth", []), label_map
                )
            # Persist scored results so local file and Hub have scores (notebook 3 can pull and analyze)
            save_results_to_disk(self.config, results)
            push_results_to_hub(self.config, results)
        return results

    def __call__(
        self,
        samples: list[dict],
        *,
        score: bool = True,
        conditions: list[str] | None = None,
        merge_with_saved: bool = True,
        run_type: str | None = None,
    ) -> list[dict]:
        """Alias for run(); allows runner(samples) or runner(samples, conditions=[\"with_skills\"], run_type=\"main\")."""
        return self.run(
            samples,
            score=score,
            conditions=conditions,
            merge_with_saved=merge_with_saved,
            run_type=run_type,
        )
