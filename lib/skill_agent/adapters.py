"""Thin adapters: ModelAdapter (generate_fn), Retriever (retrieve_fn)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


class ModelAdapter:
    """
    Implement this using Transformers directly.
    Keep it format-only: messages/tools -> text.
    """

    def __init__(self, generate_fn: Callable[..., str]):
        self._generate_fn = generate_fn

    def generate(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Callable]] = None
    ) -> str:
        return self._generate_fn(messages=messages, tools=tools)


class Retriever:
    """
    Implement however you like (TF-IDF, BM25, vectors).
    Must be deterministic for ablation fairness.
    """

    def __init__(
        self, retrieve_fn: Callable[[str, int], Tuple[str, List[Dict[str, Any]]]]
    ):
        self._retrieve_fn = retrieve_fn

    def retrieve(self, query: str, k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
        # returns (context_string, retrieved_docs_metadata)
        return self._retrieve_fn(query, k)
