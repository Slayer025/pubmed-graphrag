"""Generate answer use case."""

from __future__ import annotations

from src.application.ports import LLMClient
from src.domain.entities.retrieval_result import RetrievalResult
from src.domain.value_objects.query import Query


class GenerateAnswerUseCase:
    """Generate an answer from retrieved context and a query."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def execute(self, query: Query, context: list[RetrievalResult]) -> str:
        """Return generated answer string."""
        prompt = self._build_prompt(query.text, context)
        return self.llm.complete(prompt)

    @staticmethod
    def _build_prompt(query: str, context: list[RetrievalResult]) -> str:
        parts = [
            "You are a biomedical research assistant. Answer the question using only the context below.\n",
            "Context:\n",
        ]
        for rank, result in enumerate(context, start=1):
            parts.append(
                f"[{rank}] chunk_id={result.chunk_id} article_id={result.article_id} "
                f"combined_score={result.combined_score:.4f}\n{result.text}\n"
            )
        parts.append(f"\nQuestion: {query}\n\nAnswer:")
        return "\n".join(parts)
