"""Tool-call parsing: <tool_call>, JSON, harness bracket syntax, parse_all_tool_calls."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    m = _TOOL_CALL_TAG_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if "name" in obj and isinstance(obj.get("arguments"), dict):
                return {"name": obj["name"], "arguments": obj["arguments"]}
        except json.JSONDecodeError:
            return None

    # fallback: try first JSON object in the output
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(text[start : end + 1])
            if "name" in obj and isinstance(obj.get("arguments"), dict):
                return {"name": obj["name"], "arguments": obj["arguments"]}
            if "name" in obj and isinstance(obj.get("parameters"), dict):
                return {"name": obj["name"], "arguments": obj["parameters"]}
    except Exception:
        pass

    # Fallback: harness-style detection ([TOOL_CALL: analyze_pii], lenient intent) so all models work
    try:
        from ablation_harness.tools.detection import detect_tool_call as harness_detect
    except ImportError:
        pass
    else:
        tool_name, param = harness_detect(text)
        if tool_name:
            if tool_name == "view_skill":
                return {
                    "name": "view_skill",
                    "arguments": {"skill_name": param or "pii-detection"},
                }
            return {"name": tool_name, "arguments": {}}

    # Local bracket fallback so [TOOL_CALL: list_skills] etc. work even when ablation_harness is not importable
    stripped = text.strip()
    if re.search(r"\[TOOL_CALL\s*:\s*list_skills\s*\]", stripped, re.IGNORECASE):
        return {"name": "list_skills", "arguments": {}}
    view_m = re.search(
        r"\[TOOL_CALL\s*:\s*view_skill\s+(\w+(?:-\w+)?)\s*\]", stripped, re.IGNORECASE
    )
    if view_m:
        return {
            "name": "view_skill",
            "arguments": {"skill_name": view_m.group(1) or "pii-detection"},
        }
    if re.search(r"\[TOOL_CALL\s*:\s*analyze_pii\s*\]", stripped, re.IGNORECASE):
        return {"name": "analyze_pii", "arguments": {}}

    return None


def parse_all_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Return all bracket-style tool calls in order of first occurrence. Reduces spinning when model outputs multiple in one response."""
    stripped = text.strip()
    results: List[Tuple[int, Dict[str, Any]]] = []
    for m in re.finditer(
        r"\[TOOL_CALL\s*:\s*list_skills\s*\]", stripped, re.IGNORECASE
    ):
        results.append((m.start(), {"name": "list_skills", "arguments": {}}))
    for m in re.finditer(
        r"\[TOOL_CALL\s*:\s*view_skill\s+(\w+(?:-\w+)?)\s*\]", stripped, re.IGNORECASE
    ):
        results.append(
            (
                m.start(),
                {
                    "name": "view_skill",
                    "arguments": {"skill_name": m.group(1) or "pii-detection"},
                },
            )
        )
    for m in re.finditer(
        r"\[TOOL_CALL\s*:\s*analyze_pii\s*\]", stripped, re.IGNORECASE
    ):
        results.append((m.start(), {"name": "analyze_pii", "arguments": {}}))
    results.sort(key=lambda x: x[0])
    return [tc for _, tc in results]
