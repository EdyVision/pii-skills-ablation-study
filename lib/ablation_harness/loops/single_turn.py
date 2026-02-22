"""Single-turn dispatch for zero_shot and with_docs conditions."""

from __future__ import annotations

from ablation_harness.parsing.json_response import parse_json_response


def run_single_turn(model, prompt: str, max_tokens: int) -> dict:
    """Run a single-turn LLM call and parse the JSON response.

    Used for ``zero_shot`` and ``with_docs`` conditions.
    """
    turns = 0
    turns += 1  # one model generate
    response = model.generate(prompt, max_tokens=max_tokens)
    predictions = parse_json_response(response)
    return {
        "predictions": predictions,
        "raw_response": response,
        "tool_executed": False,
        "skill_viewed": False,
        "conversation_turns": turns,
        "error": None,
    }
