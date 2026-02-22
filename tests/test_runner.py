"""Tests for run_experiments — CUDA guard and entrypoint behavior."""

import pytest

from ablation_harness.config import HarnessConfig
from ablation_harness.runner import run_experiments
from ablation_harness.tools.registry import ToolRegistry


class TestRunExperimentsCudaGuard:
    """CUDA path is not yet validated; run_experiments must raise when hardware is cuda."""

    def test_cuda_raises_not_implemented_error(self):
        config = HarnessConfig(hardware="cuda")
        prompts = {}
        registry = ToolRegistry()
        with pytest.raises(NotImplementedError) as exc_info:
            run_experiments(config, [], prompts, registry, merge_with_saved=False)
        assert "CUDA" in str(exc_info.value)
        assert "Future research" in str(exc_info.value)
