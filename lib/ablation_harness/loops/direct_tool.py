"""Direct tool loop for the with_tools condition."""

from __future__ import annotations

import json

from ablation_harness.config import HarnessConfig
from ablation_harness.parsing.json_response import parse_json_response
from ablation_harness.tools.detection import check_for_tool_call
from ablation_harness.tools.registry import ToolRegistry


def run_direct_tool_loop(
    model,
    sample: dict,
    prompt: str,
    config: HarnessConfig,
    tool_registry: ToolRegistry,
) -> dict:
    """Multi-turn agentic loop for with_tools condition (direct tool access, no discovery)."""
    turns = 0

    turns += 1
    response1 = model.generate(prompt, max_tokens=config.max_tokens)

    tool_called = check_for_tool_call(response1)

    if tool_called and tool_registry.has_tool("analyze_pii"):
        tool_result = tool_registry.execute("analyze_pii", text=sample["text"])

        turn2_prompt = (
            "You called the analyze_pii tool. Here are the results:\n\n"
            f"```json\n{json.dumps(tool_result, indent=2)}\n```\n\n"
            "Now format your final answer as a JSON array. Each PII item should have: "
            "type, text, start, end.\nOnly output the JSON array, nothing else."
        )

        turns += 1
        response2 = model.generate(turn2_prompt, max_tokens=config.max_tokens)

        predictions = parse_json_response(response2)

        if not predictions:
            predictions = tool_result.get("detections", [])

        return {
            "predictions": predictions,
            "raw_response": f"[Turn 1]\n{response1}\n\n[Turn 2]\n{response2}",
            "tool_executed": True,
            "skill_viewed": False,
            "conversation_turns": turns,
            "error": None,
        }
    else:
        predictions = parse_json_response(response1)
        return {
            "predictions": predictions,
            "raw_response": response1,
            "tool_executed": False,
            "skill_viewed": False,
            "conversation_turns": turns,
            "error": None,
        }
