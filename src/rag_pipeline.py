"""RAG pipeline combining graph-enhanced retrieval with optional generation.

Phase 3 implements retrieval only. ``generate()`` is a placeholder interface
with a clear contract for future OpenAI/Ollama integration.

Phase 5 adds optional query decomposition and graph-based re-ranking without
changing the default behavior of ``generate()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from src.config import AppConfig
from src.retriever import RetrievalResult, Retriever, create_retriever

if TYPE_CHECKING:
    from src.graph_reranker import GraphReranker
    from src.query_decomposer import QueryDecomposer

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for future LLM generation backends (OpenAI, Ollama, etc.)."""

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a text completion for the given prompt."""
        ...


class MockLLMClient:
    """Placeholder LLM that echoes the prompt context.

    Useful for testing the RAG pipeline without external API keys.
    """

    def __init__(self, max_chars: int = 500) -> None:
        self.max_chars = max_chars

    def complete(self, prompt: str, **kwargs: Any) -> str:
        return (
            "[MOCK LLM] I would answer based on the retrieved context.\n\n"
            f"Prompt preview:\n{prompt[:self.max_chars]}..."
        )


@dataclass(frozen=True)
class RAGResponse:
    """Output of a single RAG query."""

    query: str
    context: list[RetrievalResult]
    answer: str


class RAGPipeline:
    """End-to-end RAG interface: retrieve, then generate."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient | None = None,
        decomposer: QueryDecomposer | None = None,
        reranker: GraphReranker | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or MockLLMClient()
        self.decomposer = decomposer
        self.reranker = reranker

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Return ranked context chunks for the query."""
        return self.retriever.retrieve(query)

    def _apply_reranker(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Apply the optional graph reranker."""
        if self.reranker is None:
            return results
        return self.reranker.rerank(query, results)

    def retrieve_reranked(self, query: str) -> list[RetrievalResult]:
        """Retrieve and optionally apply graph re-ranking."""
        results = self.retrieve(query)
        return self._apply_reranker(query, results)

    def retrieve_decomposed(
        self,
        query: str,
        *,
        apply_reranker: bool = True,
    ) -> tuple[list[str], list[RetrievalResult]]:
        """Retrieve for the original query and any LLM-decomposed sub-queries.

        Returns the list of sub-queries used and the merged, ranked results.
        """
        if self.decomposer is None or not self.decomposer.config.enabled:
            return [query], self.retrieve_reranked(query)

        sub_queries = self.decomposer.decompose(query)
        if len(sub_queries) <= 1:
            return sub_queries, self.retrieve_reranked(query)

        logger.info("Retrieving for %d sub-queries.", len(sub_queries))
        best_by_chunk: dict[str, RetrievalResult] = {}

        for sub_query in sub_queries:
            sub_results = self.retriever.retrieve(sub_query)
            if apply_reranker and self.reranker is not None:
                sub_results = self.reranker.rerank(sub_query, sub_results)
            for result in sub_results:
                existing = best_by_chunk.get(result.chunk_id)
                if existing is None or result.combined_score > existing.combined_score:
                    best_by_chunk[result.chunk_id] = result

        merged = sorted(best_by_chunk.values(), key=lambda r: r.combined_score, reverse=True)
        max_results = self.retriever.config.retrieval.max_results
        return sub_queries, merged[:max_results]

    def _build_prompt(self, query: str, context: list[RetrievalResult]) -> str:
        """Build a grounded QA prompt from retrieved chunks."""
        prompt_parts = [
            "You are a biomedical research assistant. Answer the question using only the context below.\n",
            "Context:\n",
        ]
        for rank, result in enumerate(context, start=1):
            prompt_parts.append(
                f"[{rank}] chunk_id={result.chunk_id} article_id={result.article_id} "
                f"combined_score={result.combined_score:.4f}\n{result.text}\n"
            )
        prompt_parts.append(f"\nQuestion: {query}\n\nAnswer:")
        return "\n".join(prompt_parts)

    def generate(
        self,
        query: str,
        context: list[RetrievalResult] | None = None,
        *,
        use_reranker: bool = True,
    ) -> RAGResponse:
        """Retrieve (if needed) and generate an answer.

        Args:
            query: User question.
            context: Optional pre-retrieved context. If None, retrieve is called.
            use_reranker: Whether to apply the optional graph reranker when
                retrieving context. Ignored when ``context`` is provided.

        Returns:
            A ``RAGResponse`` containing the query, context, and generated answer.
        """
        if context is None:
            if use_reranker and self.reranker is not None:
                context = self.retrieve_reranked(query)
            else:
                context = self.retrieve(query)

        prompt = self._build_prompt(query, context)
        logger.info("Generating answer for query: %s", query)
        answer = self.llm.complete(prompt)
        logger.info("Generated answer length: %d chars", len(answer))
        return RAGResponse(query=query, context=context, answer=answer)

    def generate_decomposed(
        self,
        query: str,
        *,
        use_reranker: bool = True,
    ) -> RAGResponse:
        """Decompose the query, retrieve per sub-query, and generate an answer."""
        sub_queries, context = self.retrieve_decomposed(
            query,
            apply_reranker=use_reranker,
        )
        logger.info(
            "Generating answer for query using %d sub-question(s).",
            len(sub_queries),
        )
        prompt = self._build_prompt(query, context)
        answer = self.llm.complete(prompt)
        return RAGResponse(query=query, context=context, answer=answer)

    def run(self, query: str) -> RAGResponse:
        """Convenience alias for ``generate``."""
        return self.generate(query)


def create_rag_pipeline(
    config: AppConfig | None = None,
    llm: LLMClient | None = None,
    decomposer: QueryDecomposer | None = None,
    reranker: GraphReranker | None = None,
) -> RAGPipeline:
    """Factory helper for building a fully configured RAG pipeline."""
    retriever = create_retriever(config)
    return RAGPipeline(
        retriever=retriever,
        llm=llm,
        decomposer=decomposer,
        reranker=reranker,
    )
