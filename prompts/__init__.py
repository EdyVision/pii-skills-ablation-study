"""Prompt templates for PII detection experiments."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load_prompt(condition: str) -> str:
    """Load a prompt template by condition name.

    Args:
        condition: One of 'zero_shot', 'with_docs', 'with_skills'

    Returns:
        Prompt template string with {text} placeholder
    """
    prompt_file = PROMPTS_DIR / f"{condition}.txt"
    if not prompt_file.exists():
        raise ValueError(
            f"Unknown condition: {condition}. Available: zero_shot, with_docs, with_skills"
        )
    return prompt_file.read_text()


def format_prompt(condition: str, text: str) -> str:
    """Format a prompt with the given text.

    Args:
        condition: One of 'zero_shot', 'with_docs', 'with_skills'
        text: The text to analyze for PII

    Returns:
        Formatted prompt ready for LLM
    """
    template = load_prompt(condition)
    return template.format(text=text)


# Token counts for cost estimation (approximate)
PROMPT_TOKEN_ESTIMATES = {
    "zero_shot": 150,
    "with_docs": 350,
    "with_skills": 800,
}
