"""CLI for running ablation-harness experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ablation_harness import HarnessConfig, load_prompts, run_experiments
from ablation_harness.config import detect_hardware
from ablation_harness.scoring.metrics import compute_metrics
from ablation_harness.scoring.normalize import load_label_map
from ablation_harness.tools.registry import ToolRegistry, PiiCodexTool

try:
    from pii_codex.services.analysis_service import PIIAnalysisService
except ImportError:
    PIIAnalysisService = None


def _resolve_config_paths(config: HarnessConfig, repo_root: Path) -> None:
    """Resolve relative paths in config against repo_root."""
    for key in ("prompts_dir", "skills_dir", "results_dir", "label_map_path"):
        val = getattr(config, key, None)
        if val is not None and not Path(val).is_absolute():
            setattr(config, key, repo_root / val)


def main(
    argv: list[str] | None = None,
    *,
    repo_root: Path | None = None,
) -> None:
    """Run experiments from the command line."""
    repo_root = repo_root or Path.cwd()
    parser = argparse.ArgumentParser(
        description="Run PII skills ablation experiments (ablation_harness CLI)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config (default: auto from detected hardware → config/<mlx|cuda>.yaml)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Path to JSON file with samples (list of dicts with 'id', 'text', 'source', 'ground_truth')",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Override conditions to run (e.g. zero_shot with_docs)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repo root for resolving relative config/samples paths (default: cwd)",
    )
    args = parser.parse_args(argv)

    if args.repo is not None:
        repo_root = Path(args.repo).resolve()
    config_dir = repo_root / "config"

    if args.config is not None:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        config = HarnessConfig.from_yaml(config_path)
        print(f"Config: {config_path.name} (explicit)")
    else:
        config = HarnessConfig.from_detected(config_dir)
        print(f"Config: {detect_hardware()}.yaml (detected)")
    _resolve_config_paths(config, repo_root)
    if args.conditions:
        config.conditions = args.conditions

    samples_path = Path(args.samples)
    if not samples_path.is_absolute():
        samples_path = repo_root / samples_path
    with open(samples_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {samples_path}")

    prompts = load_prompts(config)
    print(f"Loaded {len(prompts)} prompt templates")

    registry = ToolRegistry()
    if PIIAnalysisService is not None:
        analyzer = PIIAnalysisService()
        registry.register(PiiCodexTool(analyzer))
        print("PII-Codex tool registered")
    else:
        print(
            "PII-Codex not available — with_tools/with_skills will run without tool execution"
        )

    # with_tools / with_skills require a skill runner. AblationRunner sets this automatically;
    # the bare CLI must too, or those conditions raise. (zero_shot/with_docs don't need it.)
    if getattr(config, "with_skills_runner", None) is None:
        try:
            from skill_agent import run_skill_agent

            config.with_skills_runner = run_skill_agent
            print("Skill runner wired (with_tools/with_skills enabled)")
        except ImportError as e:
            print(
                f"Warning: skill_agent unavailable ({e}); with_tools/with_skills will fail. "
                "Install deps or run only --conditions zero_shot with_docs."
            )

    results = run_experiments(config, samples, prompts, registry)

    if config.label_map_path and config.label_map_path.exists():
        label_map = load_label_map(config.label_map_path)
        for r in results:
            r["scores"] = compute_metrics(
                r["predictions"], r.get("ground_truth", []), label_map
            )
        print(f"Scored {len(results)} results")
        # Persist the scored results so the local file has scores too (CLI parity with AblationRunner).
        from ablation_harness.runner import save_results_to_disk

        save_results_to_disk(config, results)

    print(f"Done. {len(results)} results saved.")


if __name__ == "__main__":
    main()
