"""Extract structured JSON arrays from model output."""

from __future__ import annotations

import json
import re


def parse_json_response(response: str) -> list[dict]:
    """Extract a JSON array of PII detections from a model response.

    Tries markdown fenced blocks first, then bare ``[…]`` arrays.  Each item
    must be a dict containing a ``"type"`` key to be considered valid.
    """
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"(\[[\s\S]*?\])",
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    valid_items = [
                        item
                        for item in parsed
                        if isinstance(item, dict) and "type" in item
                    ]
                    return valid_items
            except json.JSONDecodeError:
                continue

    return []
