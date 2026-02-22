"""ToolRegistry: wrap callable tools and execute by name + arguments."""

from __future__ import annotations

from typing import Any, Callable, Dict, List


class ToolRegistry:
    """
    Wrap your callable tools and enforce canonical calling.
    """

    def __init__(self, tools: List[Callable[..., Any]]):
        self.tools = tools
        self._by_name = {t.__name__: t for t in tools}

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        if name not in self._by_name:
            return f"TOOL_ERROR: unknown tool '{name}'"
        fn = self._by_name[name]
        try:
            out = fn(**arguments)
            return str(out)
        except TypeError as e:
            return f"TOOL_ERROR(TypeError): {e}"
        except Exception as e:
            return f"TOOL_ERROR(Exception): {e}"
