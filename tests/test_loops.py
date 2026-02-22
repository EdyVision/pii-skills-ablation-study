"""Tests for experiment loops — skill discovery completion and timeout handling."""

import pytest
from unittest.mock import MagicMock

from ablation_harness.config import HarnessConfig
from ablation_harness.loops.skill_discovery import (
    run_skill_discovery_loop,
    _build_result,
)
from ablation_harness.loops.direct_tool import run_direct_tool_loop
from ablation_harness.tools.registry import ToolRegistry


def _make_config(**overrides) -> HarnessConfig:
    defaults = dict(
        hardware="cpu",
        max_tokens=100,
        max_turns=5,
        max_seconds_per_sample=60,
        debug=False,
        verbose_skill_loop=False,
        skills_dir=pytest.importorskip("pathlib").Path("/nonexistent"),
    )
    defaults.update(overrides)
    return HarnessConfig(**defaults)


class TestBuildResult:

    def test_basic_structure(self):
        result = _build_result(
            [("user", "hello"), ("assistant", "world")],
            [{"type": "PERSON"}],
            True,
            True,
            2,
            None,
        )
        assert result["predictions"] == [{"type": "PERSON"}]
        assert result["tool_executed"] is True
        assert result["skill_viewed"] is True
        assert result["conversation_turns"] == 2
        assert result["error"] is None
        assert "[user]" in result["raw_response"]

    def test_error_result(self):
        result = _build_result([], [], False, False, 1, "timeout")
        assert result["error"] == "timeout"
        assert result["predictions"] == []


class TestDirectToolLoop:

    def test_tool_called_and_executed(self):
        model = MagicMock()
        model.generate.side_effect = [
            "[TOOL_CALL: analyze_pii]",
            '```json\n[{"type": "PERSON", "text": "John", "start": 0, "end": 4}]\n```',
        ]
        registry = ToolRegistry()
        tool = MagicMock()
        tool.name = "analyze_pii"
        tool.execute.return_value = {
            "detections": [{"type": "PERSON", "text": "John", "start": 0, "end": 4}],
            "sanitized_text": "[PERSON]",
            "sanitized_text_context": "[PERSON]",
        }
        registry.register(tool)

        config = _make_config()
        sample = {"text": "Contact John at john@email.com"}
        result = run_direct_tool_loop(model, sample, "prompt", config, registry)

        assert result["tool_executed"] is True
        assert len(result["predictions"]) == 1

    def test_no_tool_call(self):
        model = MagicMock()
        model.generate.return_value = (
            '```json\n[{"type": "PERSON", "text": "John", "start": 0, "end": 4}]\n```'
        )
        registry = ToolRegistry()
        config = _make_config()
        sample = {"text": "Contact John"}
        result = run_direct_tool_loop(model, sample, "prompt", config, registry)

        assert result["tool_executed"] is False
        assert len(result["predictions"]) == 1


class TestSkillDiscoveryLoop:

    def test_completes_in_expected_turns(self):
        """Model follows the happy path: list_skills -> view_skill -> analyze_pii -> final answer."""
        call_count = 0

        def mock_generate(prompt, max_tokens):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "[TOOL_CALL: list_skills]"
            elif call_count == 2:
                return "[TOOL_CALL: view_skill pii-detection]"
            elif call_count == 3:
                return "[TOOL_CALL: analyze_pii]"
            else:
                return '```json\n[{"type": "PERSON", "text": "John", "start": 0, "end": 4}]\n```'

        model = MagicMock()
        model.generate.side_effect = mock_generate

        registry = ToolRegistry()
        tool = MagicMock()
        tool.name = "analyze_pii"
        tool.execute.return_value = {
            "detections": [{"type": "PERSON", "text": "John", "start": 0, "end": 4}],
            "sanitized_text": "[PERSON]",
            "sanitized_text_context": "[PERSON]",
        }
        registry.register(tool)

        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skill_dir = skills_dir / "pii-detection"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "# pii-detection skill\nCall analyze_pii."
            )

            config = _make_config(skills_dir=skills_dir)
            sample = {"text": "Contact John at john@email.com"}
            result = run_skill_discovery_loop(
                model, sample, "Detect PII: {text}", config, registry
            )

        assert result["tool_executed"] is True
        assert result["skill_viewed"] is True
        assert result["conversation_turns"] <= 5
        assert result["error"] is None

    def test_timeout(self):
        """Loop should return timeout error when time limit is exceeded."""
        import time

        def slow_generate(prompt, max_tokens):
            time.sleep(0.05)
            return "I'm thinking about it..."

        model = MagicMock()
        model.generate.side_effect = slow_generate

        registry = ToolRegistry()
        config = _make_config(max_seconds_per_sample=0.01)  # 10ms timeout

        sample = {"text": "Test text"}
        result = run_skill_discovery_loop(model, sample, "Detect PII", config, registry)

        assert result["error"] == "timeout"
