"""Skill discovery loop for the with_skills condition."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ablation_harness.config import HarnessConfig
from ablation_harness.parsing.json_response import parse_json_response
from ablation_harness.tools.detection import check_for_tool_call, detect_tool_call
from ablation_harness.tools.registry import ToolRegistry


def load_skill_content(config: HarnessConfig, skill_name: str) -> str:
    """Load skill documentation from the skills directory."""
    skill_path = config.skills_dir / skill_name / "SKILL.md"
    if skill_path.exists():
        return skill_path.read_text()
    return "Call [TOOL_CALL: analyze_pii] to detect PII. Output JSON detections then sanitized text."


def execute_skill_tool(
    tool_name: str,
    config: HarnessConfig,
    tool_registry: ToolRegistry,
    param: str | None = None,
    sample_text: str | None = None,
) -> dict:
    """Execute skill discovery tools or analyze_pii via the registry."""
    available_skills = (
        [
            d.name
            for d in config.skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        ]
        if config.skills_dir.exists()
        else ["pii-detection"]
    )

    if tool_name == "list_skills":
        return {
            "tool": "list_skills",
            "result": available_skills,
            "message": f"Available skills: {', '.join(available_skills)}",
        }

    elif tool_name == "view_skill":
        if param in available_skills or param == "pii-detection":
            content = load_skill_content(config, param)
            return {
                "tool": "view_skill",
                "skill": param,
                "content": content,
                "message": "Skill loaded. Call analyze_pii next.",
            }
        else:
            return {
                "tool": "view_skill",
                "skill": param,
                "error": f"Skill '{param}' not found. Available: {', '.join(available_skills)}",
            }

    elif tool_name == "analyze_pii":
        if not tool_registry.has_tool("analyze_pii"):
            return {"tool": "analyze_pii", "error": "PII-Codex not available"}
        if sample_text is None:
            return {"tool": "analyze_pii", "error": "No text provided"}
        return tool_registry.execute("analyze_pii", text=sample_text)

    return {"error": f"Unknown tool: {tool_name}"}


def run_skill_discovery_loop(
    model,
    sample: dict,
    prompt: str,
    config: HarnessConfig,
    tool_registry: ToolRegistry,
) -> dict:
    """Multi-turn agentic loop for skills condition with skill discovery."""
    conversation_history: list[tuple[str, str]] = []
    turn = 0
    list_skills_called = False
    skill_viewed = False
    tool_executed = False
    start_time = time.perf_counter()

    max_turns = config.max_turns
    max_seconds = config.max_seconds_per_sample

    # Turn 1: Initial prompt
    current_prompt = prompt
    conversation_history.append(("user", current_prompt))

    while turn < max_turns:
        turn += 1

        # Timeout check
        if max_seconds and (time.perf_counter() - start_time) > max_seconds:
            return _build_result(
                conversation_history, [], tool_executed, skill_viewed, turn, "timeout"
            )

        if config.debug and config.verbose_skill_loop:
            print(f"  [with_skills] turn {turn}", flush=True)

        # Bounded context: initial user prompt + latest system message only
        if len(conversation_history) == 1:
            prompt_for_turn = conversation_history[0][1]
        else:
            prompt_for_turn = (
                "[user]\n"
                + conversation_history[0][1]
                + "\n\n[system]\n"
                + conversation_history[-1][1]
            )

        response = model.generate(prompt_for_turn, max_tokens=config.max_tokens)
        conversation_history.append(("assistant", response))

        # Timeout check after generate
        if max_seconds and (time.perf_counter() - start_time) > max_seconds:
            return _build_result(
                conversation_history, [], tool_executed, skill_viewed, turn, "timeout"
            )

        # Detect tool call
        tool_name, param = detect_tool_call(response)
        if tool_name is None and check_for_tool_call(response):
            tool_name, param = "analyze_pii", None

        if tool_name is None and config.debug and config.verbose_skill_loop:
            snippet = (
                response[:250] + "..." if len(response) > 250 else response
            ).replace("\n", " ")
            print(f"  [no tool] turn {turn} reply: {snippet!r}", flush=True)

        # Handle tool calls
        if tool_name == "list_skills":
            list_skills_called = True
            tool_result = execute_skill_tool("list_skills", config, tool_registry)
            system_msg = (
                f"Skills available: {', '.join(tool_result['result'])}. "
                f"Next: [TOOL_CALL: view_skill pii-detection]"
            )
            conversation_history.append(("system", system_msg))

        elif tool_name == "view_skill":
            skill_viewed = True
            tool_result = execute_skill_tool("view_skill", config, tool_registry, param)
            content = tool_result.get("content", tool_result.get("error", "Not found"))
            system_msg = (
                f"{content}\n\nNow call [TOOL_CALL: analyze_pii] to analyze the text."
            )
            conversation_history.append(("system", system_msg))

        elif tool_name == "analyze_pii":
            tool_result = execute_skill_tool(
                "analyze_pii", config, tool_registry, sample_text=sample["text"]
            )
            tool_executed = True

            if "error" in tool_result:
                system_msg = (
                    f"analyze_pii error: {tool_result['error']}. "
                    f"Complete the task without the tool."
                )
                conversation_history.append(("system", system_msg))
            else:
                detections = tool_result.get("detections", [])
                sanitized = tool_result.get(
                    "sanitized_text_context", tool_result.get("sanitized_text", "")
                )
                system_msg = (
                    f"Detections:\n```json\n{json.dumps(detections, indent=2)}\n```\n\n"
                    f"Sanitized text:\n```\n{sanitized}\n```\n\n"
                    f"Output the JSON array of detections, then the sanitized text."
                )
                conversation_history.append(("system", system_msg))

        else:
            # No tool call — check for final answer or nudge
            predictions = parse_json_response(response)

            if turn >= max_turns:
                if not predictions and tool_registry.has_tool("analyze_pii"):
                    tool_result = tool_registry.execute(
                        "analyze_pii", text=sample["text"]
                    )
                    predictions = tool_result.get("detections", [])
                    if predictions:
                        tool_executed = True
                return _build_result(
                    conversation_history,
                    predictions or [],
                    tool_executed,
                    skill_viewed,
                    turn,
                    None,
                )

            if predictions and tool_executed:
                return _build_result(
                    conversation_history,
                    predictions,
                    tool_executed,
                    skill_viewed,
                    turn,
                    None,
                )

            # Nudge
            if predictions and not tool_executed:
                system_msg = "You must call analyze_pii before answering. Reply with: [TOOL_CALL: analyze_pii]"
            elif not list_skills_called:
                system_msg = "Reply with exactly: [TOOL_CALL: list_skills]"
            elif not skill_viewed:
                system_msg = "Reply with exactly: [TOOL_CALL: view_skill pii-detection]"
            elif not tool_executed:
                system_msg = "Reply with exactly: [TOOL_CALL: analyze_pii]"
            else:
                system_msg = "Output your final answer: JSON array of detections, then sanitized text."

            conversation_history.append(("system", system_msg))

    # Max turns reached: use model output or fall back to tool so we never return empty
    predictions = parse_json_response(conversation_history[-1][1])
    if not predictions and tool_registry.has_tool("analyze_pii"):
        tool_result = tool_registry.execute("analyze_pii", text=sample["text"])
        predictions = tool_result.get("detections", [])
        if predictions:
            tool_executed = True  # record that we used tool output

    return _build_result(
        conversation_history,
        predictions or [],
        tool_executed,
        skill_viewed,
        turn,
        None,
    )


def _build_result(
    conversation_history: list[tuple[str, str]],
    predictions: list,
    tool_executed: bool,
    skill_viewed: bool,
    turns: int,
    error: str | None,
) -> dict:
    """Standard result dict for skill discovery loop."""
    return {
        "predictions": predictions,
        "raw_response": "\n\n".join(
            [f"[{role}]\n{content}" for role, content in conversation_history]
        ),
        "tool_executed": tool_executed,
        "skill_viewed": skill_viewed,
        "conversation_turns": turns,
        "error": error,
    }
