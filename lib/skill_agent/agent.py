"""
SkillsAgent: LangGraph orchestration for skillpack + RAG + tools.

One graph, configurable via RunConfig. Pluggable adapters: model.generate(), retriever.retrieve(), tool_registry.execute().
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import StateGraph, END

from skill_agent.adapters import ModelAdapter, Retriever
from skill_agent.config import AgentState, RunConfig
from skill_agent.registry import ToolRegistry
from skill_agent.tool_parsing import parse_all_tool_calls, parse_tool_call


class SkillsAgent:
    def __init__(
        self,
        model: ModelAdapter,
        tools: List[Callable[..., Any]],
        skillpack_text: str = "",
        retriever: Optional[Retriever] = None,
        rag_top_k: int = 4,
    ):
        self.model = model
        self.tool_registry = ToolRegistry(tools)
        self.tools = tools  # passed to model adapter for tool-aware templates
        self.skillpack_text = skillpack_text
        self.retriever = retriever
        self.rag_top_k = rag_top_k

        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(AgentState)

        # Nodes
        g.add_node("init", self._init)
        g.add_node("prepare_context", self._prepare_context)
        g.add_node("model_step", self._model_step)
        g.add_node("tool_route", self._tool_route)
        g.add_node("execute_tool", self._execute_tool)
        g.add_node("finalize", self._finalize)

        # Edges
        g.set_entry_point("init")
        g.add_edge("init", "prepare_context")
        g.add_edge("prepare_context", "model_step")
        g.add_edge("model_step", "tool_route")

        g.add_conditional_edges(
            "tool_route",
            self._tool_route_decision,
            {
                "do_tool": "execute_tool",
                "final": "finalize",
            },
        )

        g.add_conditional_edges(
            "execute_tool",
            self._after_execute_decision,
            {"execute_tool": "execute_tool", "model_step": "model_step"},
        )
        g.add_edge("finalize", END)

        return g.compile()

    def _init(self, state: AgentState) -> AgentState:
        state.setdefault("messages", [])
        state.setdefault("tool_trace", [])
        state.setdefault("retrieved_docs", [])
        state["tool_step_count"] = 0
        state["pending_tool_call"] = None
        state["pending_tool_calls"] = []
        state["final"] = ""
        return state

    def _prepare_context(self, state: AgentState) -> AgentState:
        task = state["task"]
        cfg = state["run_config"]

        base_system = (
            "You are an agent. Be precise and follow provided instructions.\n"
            "If tools are available and needed, call them.\n"
        )

        # Skillpack injection (PII Codex goes here)
        skill_block = ""
        if cfg.use_skillpack and self.skillpack_text.strip():
            skill_block = (
                "\n=== SKILL PACK ===\n"
                f"{self.skillpack_text.strip()}\n"
                "==================\n"
            )

        # Retrieval injection
        retrieved_context = ""
        retrieved_docs: List[Dict[str, Any]] = []
        if cfg.use_rag and self.retriever is not None:
            retrieved_context, retrieved_docs = self.retriever.retrieve(
                task, k=self.rag_top_k
            )
            if retrieved_context.strip():
                retrieved_context = (
                    "\n=== REFERENCES (retrieved) ===\n"
                    f"{retrieved_context.strip()}\n"
                    "==============================\n"
                )

        system_prompt = base_system + skill_block + retrieved_context

        state["system_prompt"] = system_prompt
        state["retrieved_context"] = retrieved_context
        state["retrieved_docs"] = retrieved_docs

        # Reset messages fresh per run (recommended for ablation clarity)
        state["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]
        return state

    def _model_step(self, state: AgentState) -> AgentState:
        cfg = state["run_config"]

        # If tools are disabled for this condition, do not pass tools into the template.
        tools_for_model = self.tools if cfg.use_tools else None

        text = self.model.generate(messages=state["messages"], tools=tools_for_model)
        state["messages"].append({"role": "assistant", "content": text})
        return state

    def _tool_route(self, state: AgentState) -> AgentState:
        cfg = state["run_config"]
        last = state["messages"][-1]["content"]

        if not cfg.use_tools:
            state["pending_tool_call"] = None
            state["pending_tool_calls"] = []
            return state

        # Stop if max tool loops exceeded
        if state["tool_step_count"] >= cfg.max_tool_steps:
            state["pending_tool_call"] = None
            state["pending_tool_calls"] = []
            return state

        all_calls = parse_all_tool_calls(last)
        if not all_calls:
            single = parse_tool_call(last)
            if single:
                all_calls = [single]
        if os.environ.get("PII_ABLATION_DEBUG") and cfg.use_tools:
            step = state["tool_step_count"]
            print(
                f"[pii_abl] turn (after {step} tools) repr:",
                repr(last[:500]),
                "| parse_all_tool_calls:",
                all_calls,
            )
        # Truncate assistant message at first tool call for all models (same method across the board)
        if all_calls:
            first_match = re.search(
                r"\[TOOL_CALL\s*:\s*(?:list_skills|view_skill\s+\w+(?:-\w+)?|analyze_pii)\s*\]",
                last,
                re.IGNORECASE,
            )
            if first_match:
                end = first_match.end()
                newline = last.find("\n", end)
                if newline != -1:
                    end = newline + 1
                state["messages"][-1]["content"] = last[:end]
        state["pending_tool_call"] = all_calls[0] if all_calls else None
        state["pending_tool_calls"] = all_calls
        return state

    def _tool_route_decision(self, state: AgentState) -> str:
        return "do_tool" if state.get("pending_tool_calls") else "final"

    def _after_execute_decision(self, state: AgentState) -> str:
        return "execute_tool" if state.get("pending_tool_calls") else "model_step"

    def _execute_tool(self, state: AgentState) -> AgentState:
        pending = state.get("pending_tool_calls") or []
        if not pending:
            return state

        tc = pending[0]
        state["pending_tool_calls"] = pending[1:]
        state["pending_tool_call"] = None

        name = tc["name"]
        args = tc["arguments"]

        result = self.tool_registry.execute(name, args)
        state["tool_step_count"] = state.get("tool_step_count", 0) + 1

        # Return new list so LangGraph state merge keeps the update (in-place append may not)
        trace = list(state.get("tool_trace", []))
        trace.append({"name": name, "arguments": args, "result": result})
        state["tool_trace"] = trace

        # Append tool result message (name so adapter can send the right follow-up prompt)
        state["messages"] = list(state.get("messages", [])) + [
            {"role": "tool", "content": result, "name": name}
        ]
        return state

    def _finalize(self, state: AgentState) -> AgentState:
        final = ""
        for msg in reversed(state["messages"]):
            if msg.get("role") == "assistant":
                final = msg.get("content", "").strip()
                break
        state["final"] = final
        return state

    def run(self, task: str, config: RunConfig) -> Dict[str, Any]:
        init_state: AgentState = {"task": task, "run_config": config}
        out = self.graph.invoke(init_state)

        return {
            "final": out.get("final", ""),
            "messages": out.get("messages", []),
            "tool_trace": out.get("tool_trace", []),
            "retrieved_docs": out.get("retrieved_docs", []),
            "system_prompt": out.get("system_prompt", ""),
        }
