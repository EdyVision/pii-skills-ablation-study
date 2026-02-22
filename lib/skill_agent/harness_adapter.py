"""
Bridge from the ablation harness to the LangGraph SkillsAgent.

Provides run_skill_agent(model, sample, prompt, config, tool_registry)
returning the same dict shape as run_skill_discovery_loop so the runner can score and save
without any change to the rest of the harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from ablation_harness.loops.skill_discovery import (
    execute_skill_tool,
    load_skill_content,
)
from ablation_harness.parsing.json_response import parse_json_response

from skill_agent.adapters import ModelAdapter, Retriever
from skill_agent.agent import SkillsAgent
from skill_agent.config import RunConfig

if TYPE_CHECKING:
    from ablation_harness.config import HarnessConfig
    from ablation_harness.tools.registry import ToolRegistry

# Reuse agent per (model, condition, model_name) to avoid rebuilding graph/docs/tools every sample
_agent_cache: Dict[
    Tuple[int, bool, bool, bool, str], Tuple[SkillsAgent, Dict[str, Any]]
] = {}

# Models with reliable native tool calling: pass full message list so they see assistant/tool turns
_NATIVE_TOOL_MODELS = frozenset({"llama3_8b", "qwen2_7b"})


def _build_docs(config: "HarnessConfig") -> List[str]:
    """Doc chunks for RAG (with_docs content + skill doc)."""
    docs: List[str] = []
    prompts_dir = Path(config.prompts_dir)
    with_docs_path = prompts_dir.parent / "with_docs.txt"
    if with_docs_path.exists():
        text = with_docs_path.read_text()
        chunks = [c.strip() for c in text.split("\n## ") if c.strip()]
        docs.extend(chunks)
    skill_content = load_skill_content(config, "pii-detection")
    if skill_content:
        docs.append(skill_content)
    if not docs:
        docs.append(
            "Analyze text for PII. Return a JSON array of {type, text, start, end}."
        )
    return docs


def _messages_to_prompt(messages: List[dict], first_user_content: str) -> str:
    """Format messages into the single prompt string the harness model expects.
    After tool calls, use compact follow-up. When multiple tools ran in one batch (e.g. list_skills
    then view_skill), combine their results so the model sees both before being asked for analyze_pii.
    """
    if not messages:
        return first_user_content

    # Collect trailing tool messages (order: last tool first)
    tool_msgs: List[dict] = []
    for m in reversed(messages):
        if m.get("role") != "tool":
            break
        tool_msgs.append(m)
    if tool_msgs:
        # Reverse so order is list_skills, view_skill, ... then last is most recent
        tool_msgs = list(reversed(tool_msgs))
        if any(m.get("name") == "analyze_pii" for m in tool_msgs):
            last = next(
                m for m in reversed(tool_msgs) if m.get("name") == "analyze_pii"
            )
            return (
                "You called the analyze_pii tool. Here are the results:\n\n"
                f"```json\n{last.get('content', '')}\n```\n\n"
                "Now format your final answer as a JSON array. Each PII item should have: "
                "type, text, start, end.\nOnly output the JSON array, nothing else."
            )
        parts = []
        for m in tool_msgs:
            name = m.get("name", "")
            content = m.get("content", "")
            parts.append(
                f"You called {name}. Here are the results:\n\n```json\n{content}\n```"
            )
        combined = "\n\n".join(parts)
        has_view = any(m.get("name") == "view_skill" for m in tool_msgs)
        if has_view:
            return (
                f"{combined}\n\n"
                "Your next reply must be exactly: [TOOL_CALL: analyze_pii]"
            )
        return (
            f"{combined}\n\n"
            "Your next reply must be exactly: [TOOL_CALL: view_skill pii-detection]"
        )

    for m in reversed(messages):
        if m.get("role") == "system" and m != messages[0]:
            return f"[user]\n{first_user_content}\n\n[system]\n{m.get('content', '')}"
    return first_user_content


def _build_tools(
    config: "HarnessConfig",
    tool_registry: "ToolRegistry",
    run_ctx: Dict[str, Any],
    with_skills: bool = False,
    with_tools: bool = False,
) -> List[Callable[..., Any]]:
    """Callables for LangGraph ToolRegistry. Returns tools for the condition: [] for zero_shot/with_docs, [analyze_pii] for with_tools, [list_skills, view_skill, analyze_pii] for with_skills. run_ctx['sample_text'] set per run for reuse."""

    def list_skills() -> str:
        result = execute_skill_tool("list_skills", config, tool_registry, None, None)
        return json.dumps(result)

    def view_skill(skill_name: str = "pii-detection") -> str:
        result = execute_skill_tool(
            "view_skill", config, tool_registry, skill_name, None
        )
        content = result.get("content") or result.get("error") or json.dumps(result)
        return content

    def analyze_pii(text: str = "") -> str:
        if not (text or "").strip():
            text = run_ctx.get("sample_text", "")
        if not tool_registry.has_tool("analyze_pii"):
            return json.dumps({"error": "PII-Codex not available", "detections": []})
        result = tool_registry.execute("analyze_pii", text=text or "")
        return json.dumps(result) if isinstance(result, dict) else str(result)

    list_skills.__name__ = "list_skills"
    view_skill.__name__ = "view_skill"
    analyze_pii.__name__ = "analyze_pii"

    if with_skills:
        return [list_skills, view_skill, analyze_pii]
    if with_tools:
        return [analyze_pii]
    return []


def _make_retriever(docs: List[str]):
    """Simple deterministic retriever: return concatenated docs (no TF-IDF dependency)."""

    def retrieve(query: str, k: int = 4) -> tuple:
        combined = "\n\n".join(docs[: max(1, k)])
        return combined, [{"id": i, "text": d} for i, d in enumerate(docs[:k])]

    return Retriever(retrieve)


def run_skill_agent(
    model: Any,
    sample: dict,
    prompt: str,
    config: "HarnessConfig",
    tool_registry: "ToolRegistry",
    with_skills: bool = False,
    with_docs: bool = False,
    with_tools: bool = False,
    *,
    model_name: str | None = None,
) -> dict:
    """
    Run the LangGraph SkillsAgent for one sample and return a harness-shaped result.

    When with_skills_runner is this callable, the harness uses it for every condition and
    passes these three booleans. None True => zero_shot; all True => with_skills.

    For llama3_8b and qwen2_7b we pass full message list (use_native_tool_messages) so they
    see assistant/tool turns; for others we keep single-blob prompt so Gemma/Mistral still work.
    Agent is cached per (model, condition, model_name) so we only build graph/docs/tools once per condition.
    """
    sample_text = sample.get("text", "")
    # Skill agent: single cap for all conditions
    cap = 1500
    max_tokens = min(getattr(config, "max_tokens", 2048), cap)
    tool_round_max_tokens = getattr(config, "tool_round_max_tokens", cap)
    max_tool_steps = getattr(config, "max_turns", 5)
    use_native = model_name in _NATIVE_TOOL_MODELS if model_name else False
    run_config = RunConfig(
        use_skillpack=with_skills,
        use_rag=with_docs,
        use_tools=with_tools,
        max_tool_steps=max_tool_steps,
        use_native_tool_messages=use_native,
    )

    current_model_id = id(model)
    for k in list(_agent_cache):
        if k[0] != current_model_id:
            del _agent_cache[k]
    cache_key = (current_model_id, with_skills, with_docs, with_tools, model_name or "")
    if cache_key in _agent_cache:
        agent, run_ctx = _agent_cache[cache_key]
        run_ctx["sample_text"] = sample_text
    else:
        run_ctx = {"sample_text": sample_text, "use_native_tool_messages": use_native}

        def generate_fn(messages: List[dict], tools: Any = None) -> str:
            has_tool_turn = any(m.get("role") == "tool" for m in messages)
            # Same cap for tool rounds across all models
            step_max = (
                min(max_tokens, tool_round_max_tokens) if has_tool_turn else max_tokens
            )
            if run_ctx.get("use_native_tool_messages") and hasattr(
                model, "generate_from_messages"
            ):
                try:
                    out = model.generate_from_messages(messages, max_tokens=step_max)
                    if out:
                        return out
                except Exception:
                    pass
            first_user = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                run_ctx.get("sample_text", ""),
            )
            prompt_str = _messages_to_prompt(messages, first_user)
            return model.generate(prompt_str, max_tokens=step_max)

        docs = _build_docs(config)
        tools = _build_tools(
            config,
            tool_registry,
            run_ctx,
            with_skills=with_skills,
            with_tools=with_tools,
        )
        skillpack = load_skill_content(config, "pii-detection") or ""
        retriever = _make_retriever(docs)
        adapter = ModelAdapter(generate_fn)
        agent = SkillsAgent(
            model=adapter,
            tools=tools,
            skillpack_text=skillpack,
            retriever=retriever,
            rag_top_k=4,
        )
        _agent_cache[cache_key] = (agent, run_ctx)

    try:
        out = agent.run(task=prompt, config=run_config)
    except Exception as e:
        return {
            "predictions": [],
            "raw_response": None,
            "tool_executed": False,
            "skill_viewed": False,
            "conversation_turns": 0,
            "error": str(e),
        }

    final_text = out.get("final", "")
    predictions = parse_json_response(final_text)
    tool_trace = out.get("tool_trace", [])
    tool_executed = any(t.get("name") == "analyze_pii" for t in tool_trace)
    skill_viewed = with_skills and any(
        t.get("name") == "view_skill" for t in tool_trace
    )
    messages = out.get("messages", [])
    conversation_turns = len([m for m in messages if m.get("role") == "assistant"])

    return {
        "predictions": predictions,
        "raw_response": final_text,
        "tool_executed": tool_executed,
        "skill_viewed": skill_viewed,
        "conversation_turns": conversation_turns,
        "error": None,
    }
