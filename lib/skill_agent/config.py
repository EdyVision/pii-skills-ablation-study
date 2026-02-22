"""RunConfig and AgentState for the LangGraph SkillsAgent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


@dataclass(frozen=True)
class RunConfig:
    use_skillpack: bool = True
    use_rag: bool = True
    use_tools: bool = True
    max_tool_steps: int = 6  # prevent infinite loops
    use_native_tool_messages: bool = (
        False  # when True, pass full message list (Llama/Qwen); assistant truncation applies to all
    )


class AgentState(TypedDict, total=False):
    # Inputs
    task: str
    run_config: RunConfig

    # Context
    system_prompt: str
    retrieved_context: str
    retrieved_docs: List[Dict[str, Any]]  # optional: ids/scores/chunks

    # Conversation transcript (OpenAI-ish message dicts)
    messages: List[Dict[str, Any]]

    # Tooling
    tool_step_count: int
    pending_tool_call: Optional[
        Dict[str, Any]
    ]  # single (legacy); use pending_tool_calls queue
    pending_tool_calls: List[
        Dict[str, Any]
    ]  # queue: execute all from one response in order
    tool_trace: List[Dict[str, Any]]  # [{"name","arguments","result","error"}]

    # Output
    final: str
