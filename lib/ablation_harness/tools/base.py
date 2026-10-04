"""BaseTool ABC for pluggable tool implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Abstract base class for tools that can be registered with ToolRegistry."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name used for dispatch."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        """Execute the tool and return a result dict."""
        ...
