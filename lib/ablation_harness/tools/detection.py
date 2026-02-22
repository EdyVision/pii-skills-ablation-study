"""Tool-call detection helpers — detect_tool_call and check_for_tool_call."""

from __future__ import annotations

import re


def detect_tool_call(response: str) -> tuple[str | None, str | None]:
    """Detect which tool the model is calling and extract parameters.

    Returns ``(tool_name, parameter)`` or ``(None, None)``.
    Uses strict bracket syntax first (including line-only for Llama/Qwen), then
    lenient intent patterns. Tolerates trailing punctuation and extra whitespace
    so models that output e.g. "[TOOL_CALL: list_skills]." or wrap in text are still detected.
    """
    # Normalize: strip and collapse internal whitespace for line checks
    response_stripped = response.strip()

    # Exact line match (allow trailing space/punctuation so "[TOOL_CALL: list_skills]." matches)
    for line in response.split("\n"):
        line = line.strip()
        if re.match(r"\[TOOL_CALL:\s*list_skills\s*\]\.?\s*$", line, re.IGNORECASE):
            return "list_skills", None
        if re.match(r"\[TOOL_CALL:\s*analyze_pii\s*\]\.?\s*$", line, re.IGNORECASE):
            return "analyze_pii", None
        view_strict = re.match(
            r"\[TOOL_CALL:\s*view_skill\s+(\w+(?:-\w+)?)\s*\]\.?\s*$",
            line,
            re.IGNORECASE,
        )
        if view_strict:
            return "view_skill", view_strict.group(1)

    # Bracket anywhere in response (allow optional spaces around colon and inside brackets)
    patterns = [
        (r"\[TOOL_CALL\s*:\s*list_skills\s*\]", "list_skills", None),
        (r"\[TOOL_CALL\s*:\s*view_skill\s+(\w+(?:-\w+)?)\s*\]", "view_skill", 1),
        (r"\[TOOL_CALL\s*:\s*analyze_pii\s*\]", "analyze_pii", None),
    ]
    for pattern, tool_name, group_idx in patterns:
        match = re.search(pattern, response_stripped, re.IGNORECASE)
        if match:
            param = match.group(group_idx) if group_idx else None
            return tool_name, param

    # Lenient: intent phrasing (Llama/Qwen often paraphrase instead of exact bracket)
    if re.search(
        r"\blist_skills\b|list_skills\s*\(|list\s+skills|call\s+list_skills"
        r"|I will\s+list|listing\s+skills|I'll\s+list\s+skills|I\s+will\s+list\s+skills"
        r"|here\s+is\s+.*list_skills|to\s+list\s+skills|use\s+list_skills"
        r"|reply\s+with\s+.*list_skills|output\s+.*\[TOOL_CALL",
        response_stripped,
        re.IGNORECASE,
    ):
        return "list_skills", None

    view_match = re.search(
        r'view_skill\s*\(\s*["\']?(\w+(?:-\w+)?)["\']?\s*\)'
        r"|view\s+(?:the\s+)?(?:skill\s+)?[\"']?(pii-detection|\w+)[\"']?"
        r"|view_skill\s+(pii-detection|\w+)",
        response_stripped,
        re.IGNORECASE,
    )
    if view_match:
        param = next((g for g in view_match.groups() if g), None)
        return "view_skill", param or "pii-detection"
    if re.search(
        r"view_skill|view\s+the\s+skill|view\s+pii-detection",
        response_stripped,
        re.IGNORECASE,
    ):
        return "view_skill", "pii-detection"

    # Lenient: analyze_pii intent
    if re.search(
        r"analyze_pii\s*\(\)|I will use.*analyze_pii|calling.*analyze_pii"
        r"|use the analyze_pii tool|call\s+analyze_pii|invoke\s+analyze_pii"
        r"|run\s+analyze_pii|I need to.*analyze_pii|let me.*analyze_pii"
        r"|will call.*analyze_pii|to get (?:the )?(?:detections|sanitized).*analyze_pii"
        r"|\banalyze_pii\b|execute\s+analyze_pii|running\s+analyze_pii"
        r"|use the (?:pii )?detection tool|run the (?:pii )?detection tool"
        r"|execute the (?:pii )?detection tool|analyze the text"
        r"|I will (?:use|run|call) the (?:pii |detection )?tool",
        response_stripped,
        re.IGNORECASE,
    ):
        return "analyze_pii", None

    return None, None


def check_for_tool_call(response: str) -> bool:
    """Check if model is requesting the analyze_pii tool (for with_tools condition)."""
    tool_patterns = [
        r"\[TOOL_CALL:\s*analyze_pii\]",
        r"\[TOOL_CALL\].*analyze_pii",
        r"```tool_call\s*\n?\s*analyze_pii",
        r"analyze_pii\s*\(\)",
        r"\banalyze_pii\b",
        r"I will use.*analyze_pii",
        r"calling.*analyze_pii",
        r"use the analyze_pii tool",
        r"call\s+analyze_pii",
        r"invoke\s+analyze_pii",
        r"run\s+analyze_pii",
        r"I need to.*analyze_pii",
        r"let me.*analyze_pii",
        r"will call.*analyze_pii",
        r"to get (?:the )?(?:detections|sanitized).*analyze_pii",
        r"use the (?:pii )?detection tool|run the (?:pii )?detection tool|analyze the text",
    ]
    for pattern in tool_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return True
    return False
