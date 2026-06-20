"""LLM-driven query decomposition for multi-faceted biomedical questions.

The decomposer uses any ``LLMClient`` implementation from ``src.llm_client`` to
split a complex question into a list of focused sub-questions.  On any failure
it falls back to a single-element list containing the original query.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from src.application.ports import LLMClient

logger = logging.getLogger(__name__)

DEFAULT_DECOMPOSITION_PROMPT = (
    "You are a biomedical research assistant. Decompose the following question "
    "into 1 to 4 focused sub-questions that, when answered together, fully address "
    "the original question. Return ONLY a JSON array of strings, with no markdown "
    "formatting, no explanation, and no additional text.\n\n"
    "Example input: \"What are the risk factors and treatments for type 2 diabetes?\"\n"
    'Example output: ["What are the risk factors for type 2 diabetes?", '
    '"What are the treatments for type 2 diabetes?"]\n\n'
    "Question: {query}\n\n"
    "Output:"
)


@dataclass(frozen=True)
class DecomposerConfig:
    """Lightweight configuration for query decomposition."""

    enabled: bool = False
    prompt_template: str = DEFAULT_DECOMPOSITION_PROMPT
    max_sub_queries: int = 4


class QueryDecomposer:
    """Decompose a query into sub-questions using an LLM client."""

    def __init__(
        self,
        llm: LLMClient,
        config: DecomposerConfig | None = None,
    ) -> None:
        self.llm = llm
        self.config = config or DecomposerConfig()

    def decompose(self, query: str) -> list[str]:
        """Return a list of sub-questions, falling back to ``[query]``."""
        if not self.config.enabled:
            logger.debug("Decomposition disabled; using original query.")
            return [query]

        if not query or not query.strip():
            return [query]

        prompt = self.config.prompt_template.format(query=query.strip())
        try:
            raw = self.llm.complete(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM decomposition failed (%s); falling back to original query.", exc)
            return [query]

        sub_queries = self._parse_sub_queries(raw)
        if not sub_queries:
            logger.warning("No valid sub-questions parsed; falling back to original query.")
            return [query]

        if len(sub_queries) > self.config.max_sub_queries:
            logger.warning(
                "Truncating %d sub-questions to max %d.",
                len(sub_queries),
                self.config.max_sub_queries,
            )
            sub_queries = sub_queries[: self.config.max_sub_queries]

        logger.info("Decomposed query into %d sub-questions.", len(sub_queries))
        return sub_queries

    def _parse_sub_queries(self, raw: str) -> list[str]:
        """Extract a list of strings from LLM output."""
        text = (raw or "").strip()
        if not text:
            return []

        # Strip common markdown fences.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try to locate a JSON array if the response has extra text.
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            text = match.group(0)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse decomposition JSON: %s", exc)
            return []

        if isinstance(parsed, dict):
            # Some models may return {"sub_questions": [...]}.
            for key in ("sub_questions", "subqueries", "sub_queries", "questions"):
                if key in parsed and isinstance(parsed[key], list):
                    parsed = parsed[key]
                    break
            else:
                logger.warning("Decomposition JSON was a dict without a known list key.")
                return []

        if not isinstance(parsed, list):
            logger.warning("Decomposition JSON was not a list.")
            return []

        sub_queries: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                sub_queries.append(item.strip())
            elif isinstance(item, dict):
                # Some models may return {"question": "..."} objects.
                q = item.get("question") or item.get("q") or item.get("text")
                if isinstance(q, str) and q.strip():
                    sub_queries.append(q.strip())

        return sub_queries


def create_decomposer(
    llm: LLMClient | None = None,
    enabled: bool = False,
) -> QueryDecomposer:
    """Factory helper for building a decomposer."""
    config = DecomposerConfig(enabled=enabled)
    if llm is None:
        from src.llm_client import MockLLMClient

        llm = MockLLMClient()
    return QueryDecomposer(llm=llm, config=config)
