"""HarnessConfig dataclass — replaces all module-level globals."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def detect_hardware() -> str:
    """Return 'mlx', 'cuda', or 'cpu' based on platform and available backends."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class HarnessConfig:
    """Central configuration for ablation-harness experiments.

    Every function that previously read a module-level global now accepts an
    instance of this class instead.
    """

    hardware: str = "cuda"  # cuda | mlx | cpu
    max_tokens: int = 2048
    tool_round_max_tokens: int = (
        1024  # cap for tool-round generations (Llama/Qwen native path)
    )
    max_turns: int = 5
    max_seconds_per_sample: int = 120
    batch_size: int = 32
    batch_size_parallel: int = 16
    multi_turn_workers: int = 2  # 1 for mlx, 2 for cuda
    prompts_dir: Path = field(default_factory=lambda: Path("prompts/v2"))
    skills_dir: Path = field(default_factory=lambda: Path("skills"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    conditions: list[str] = field(
        default_factory=lambda: ["zero_shot", "with_docs", "with_tools", "with_skills"]
    )
    models: list[str] = field(default_factory=list)
    model_configs: dict = field(
        default_factory=dict
    )  # model_name -> {"hf": "...", "mlx": "..."}
    seed: int = 42
    debug: bool = False
    verbose_skill_loop: bool = False
    push_to_hub: bool = False
    hub_repo: str = ""
    run_type: str = "pilot"
    prompt_version: str = "v1"
    label_map_path: Path | None = None  # JSON file for PII type normalization
    with_skills_runner: Any = (
        None  # When set (callable), used for every condition; pass with_skills, with_docs, with_tools (none True => zero_shot)
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> HarnessConfig:
        """Load configuration from a flat YAML file (e.g. config/mlx.yaml)."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        # Flatten nested YAML into flat kwargs
        kwargs: dict = {}
        for key, value in data.items():
            if key in cls.__dataclass_fields__:
                kwargs[key] = value
        # Convert Path fields
        for key in ("prompts_dir", "skills_dir", "results_dir", "label_map_path"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        return cls(**kwargs)

    @classmethod
    def from_repo_config(cls, repo_root: str | Path) -> HarnessConfig:
        """Load harness config from repo root config.yaml (harness.common + harness.mlx/cuda)."""
        import yaml

        repo_root = Path(repo_root)
        config_path = repo_root / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config not found: {config_path}. "
                "Expected config.yaml with a top-level 'harness' section."
            )
        with open(config_path) as f:
            data = yaml.safe_load(f)
        harness = data.get("harness")
        if not harness or "common" not in harness:
            raise ValueError(
                "config.yaml must contain harness.common (and harness.mlx / harness.cuda)."
            )
        hardware = detect_hardware()
        common = harness["common"]
        overrides = harness.get(hardware, harness.get("cuda", {}))
        # Merge: common first, then hardware-specific overrides
        merged = {**common, **overrides}
        # Keep only HarnessConfig fields
        kwargs = {k: v for k, v in merged.items() if k in cls.__dataclass_fields__}
        for key in ("prompts_dir", "skills_dir", "results_dir", "label_map_path"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = Path(kwargs[key])
        return cls(**kwargs)

    @classmethod
    def from_detected(cls, config_dir: str | Path) -> HarnessConfig:
        """Load config: from repo config.yaml (harness section) or from config/{hardware}.yaml."""
        config_dir = Path(config_dir)
        # If config_dir is "config" subdir, repo root is parent
        repo_root = config_dir.parent if config_dir.name == "config" else config_dir
        config_yaml = repo_root / "config.yaml"
        if config_yaml.exists():
            with open(config_yaml) as f:
                import yaml

                data = yaml.safe_load(f)
            if data.get("harness") and "common" in data["harness"]:
                return cls.from_repo_config(repo_root)
        # Fallback: config/{hardware}.yaml
        hardware = detect_hardware()
        path = config_dir / f"{hardware}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Config not found: {path} (detected hardware={hardware}). "
                f"Use config.yaml with harness section, or add config/{hardware}.yaml."
            )
        return cls.from_yaml(path)
