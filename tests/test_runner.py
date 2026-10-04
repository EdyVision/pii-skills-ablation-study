"""Tests for run_experiments configuration guards."""

import pytest

from ablation_harness.config import HarnessConfig
from ablation_harness.runner import run_experiments
from ablation_harness.tools.registry import ToolRegistry


class TestSampledDecodingGuard:
    """Sampled multi-turn decoding must have deterministic execution order."""

    def test_sampled_parallel_multi_turn_raises(self):
        config = HarnessConfig(hardware="cuda", do_sample=True, multi_turn_workers=2)
        prompts = {}
        registry = ToolRegistry()
        with pytest.raises(ValueError) as exc_info:
            run_experiments(config, [], prompts, registry, merge_with_saved=False)
        assert "multi_turn_workers=1" in str(exc_info.value)
