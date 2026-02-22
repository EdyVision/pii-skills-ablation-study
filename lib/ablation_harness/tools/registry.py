"""ToolRegistry — pluggable tool dispatch replacing hardcoded execute_pii_tool calls."""

from __future__ import annotations

from ablation_harness.tools.base import BaseTool


class ToolRegistry:
    """Registry of named tools that experiment loops can call."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def execute(self, name: str, **kwargs) -> dict:
        """Execute a registered tool by name."""
        if name not in self._tools:
            return {"error": f"Unknown tool: {name}"}
        return self._tools[name].execute(**kwargs)

    def list_tools(self) -> list[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is registered."""
        return name in self._tools


class PiiCodexTool(BaseTool):
    """Wraps PII-Codex analysis as a BaseTool."""

    name = "analyze_pii"

    def __init__(self, analyzer):
        self._analyzer = analyzer

    def execute(self, text: str = None, **kwargs) -> dict:
        """Run PII-Codex analysis on *text* and return detections."""
        if text is None:
            return {"error": "No text provided"}
        try:
            result = self._analyzer.analyze_item(text)

            detections = []
            for d in result.analysis or []:
                if d.detection and d.risk_assessment:
                    detections.append(
                        {
                            "type": d.risk_assessment.pii_type_detected,
                            "text": (
                                text[d.detection.start : d.detection.end]
                                if d.detection.start and d.detection.end
                                else ""
                            ),
                            "start": d.detection.start,
                            "end": d.detection.end,
                            "risk_level": d.risk_assessment.risk_level,
                            "confidence": (
                                d.detection.score if d.detection.score else 0.0
                            ),
                        }
                    )

            sanitized_context = _context_preserving_sanitized(text, detections)
            return {
                "detections": detections,
                "sanitized_text": result.sanitized_text,
                "sanitized_text_context": sanitized_context,
                "risk_score": result.risk_score_mean,
            }
        except Exception as e:
            return {"error": str(e), "detections": []}


def _context_preserving_sanitized(text: str, detections: list) -> str:
    """Build sanitized text with [TYPE] placeholders preserving context."""
    if not detections:
        return text
    out = text
    for d in sorted(detections, key=lambda x: x["start"], reverse=True):
        start, end = d.get("start"), d.get("end")
        if start is not None and end is not None:
            pii_type = d.get("type", "PII")
            out = out[:start] + f"[{pii_type}]" + out[end:]
    return out
