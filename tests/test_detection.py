"""Tests for tool call detection — strict and lenient patterns."""

import pytest

from ablation_harness.tools.detection import check_for_tool_call, detect_tool_call


class TestDetectToolCall:
    """Tests for detect_tool_call with strict and lenient patterns."""

    # ── Strict bracket syntax ──

    def test_strict_list_skills(self):
        assert detect_tool_call("[TOOL_CALL: list_skills]") == ("list_skills", None)

    def test_strict_analyze_pii(self):
        assert detect_tool_call("[TOOL_CALL: analyze_pii]") == ("analyze_pii", None)

    def test_strict_view_skill(self):
        assert detect_tool_call("[TOOL_CALL: view_skill pii-detection]") == (
            "view_skill",
            "pii-detection",
        )

    def test_strict_in_multiline(self):
        response = "Let me check.\n[TOOL_CALL: list_skills]\nI'll review the results."
        assert detect_tool_call(response) == ("list_skills", None)

    def test_strict_analyze_in_text(self):
        response = (
            "Based on my analysis, I'll call [TOOL_CALL: analyze_pii] to detect PII."
        )
        assert detect_tool_call(response) == ("analyze_pii", None)

    # ── Lenient intent patterns ──

    def test_lenient_list_skills(self):
        assert detect_tool_call("I will list the available skills") == (
            "list_skills",
            None,
        )

    def test_lenient_analyze_pii_function_call(self):
        assert detect_tool_call("analyze_pii()") == ("analyze_pii", None)

    def test_lenient_analyze_pii_intent(self):
        assert detect_tool_call("I will use the analyze_pii tool") == (
            "analyze_pii",
            None,
        )

    def test_lenient_view_skill_with_quotes(self):
        result = detect_tool_call('view_skill("pii-detection")')
        assert result == ("view_skill", "pii-detection")

    def test_lenient_view_skill_natural(self):
        result = detect_tool_call("Let me view the skill pii-detection")
        assert result == ("view_skill", "pii-detection")

    # ── No tool call ──

    def test_no_tool_call(self):
        assert detect_tool_call("Here is my analysis of the text.") == (None, None)

    def test_no_tool_call_empty(self):
        assert detect_tool_call("") == (None, None)

    # ── Edge cases ──

    def test_case_insensitive_bracket(self):
        assert detect_tool_call("[tool_call: list_skills]") == ("list_skills", None)

    def test_line_only_bracket(self):
        """Llama outputs tool call on its own line."""
        response = "Sure, let me do that.\n[TOOL_CALL: list_skills]\n"
        assert detect_tool_call(response) == ("list_skills", None)


class TestCheckForToolCall:
    """Tests for check_for_tool_call (analyze_pii boolean check)."""

    def test_strict_bracket(self):
        assert check_for_tool_call("[TOOL_CALL: analyze_pii]") is True

    def test_function_call_syntax(self):
        assert check_for_tool_call("analyze_pii()") is True

    def test_intent_phrase(self):
        assert check_for_tool_call("I will use the analyze_pii tool") is True

    def test_no_tool(self):
        assert check_for_tool_call("Here is the PII I found:") is False

    def test_natural_language_tool(self):
        assert check_for_tool_call("run the pii detection tool") is True
