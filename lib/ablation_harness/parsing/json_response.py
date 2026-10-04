"""Decode JSON arrays without treating brackets inside strings as delimiters."""
import json
import re

def parse_json_response(response: str) -> list[dict]:
    if not isinstance(response, str):
        return []
    decoder = json.JSONDecoder()
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response, flags=re.IGNORECASE)
    for candidate in [*fenced, response]:
        for match in re.finditer(r"\[", candidate):
            try:
                parsed, _ = decoder.raw_decode(candidate[match.start():])
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, list):
                valid = [item for item in parsed if isinstance(item, dict)
                         and isinstance(item.get("type"), str) and item["type"].strip()]
                if valid or not parsed:
                    return valid
    return []
