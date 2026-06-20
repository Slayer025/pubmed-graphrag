"""Streamlit interface for the PubMed GraphRAG pipeline."""

from __future__ import annotations

import csv
import io
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Repo root (…/pubmed-graphrag), not src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if Path.cwd() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

try:
    import streamlit as st
except ImportError as exc:
    print(
        "Streamlit is not installed. Install it with: pip install streamlit\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)

from src.application.dto.search_config import SearchConfig
from src.bootstrap import bootstrap_pipeline, default_search_config
from src.domain.entities.retrieval_result import RetrievalResult
from src.infrastructure.storage.artifact_loader import ensure_deployment_artifacts
from src.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@st.cache_resource(show_spinner="Downloading deployment artifacts (once per session)...")
def _ensure_artifacts_cached() -> tuple[str, ...]:
    """Ensure remote artifacts exist on disk once per Streamlit session."""
    return ensure_deployment_artifacts()


@st.cache_resource(show_spinner="Loading PubMed GraphRAG pipeline...")
def _load_pipeline(
    llm_client_type: str,
    use_reranker: bool,
    reranker_beta: float,
    use_decomposer: bool,
) -> RAGPipeline:
    """Bootstrap the RAG pipeline once. UI retrieval config is applied per request."""
    return bootstrap_pipeline(
        llm_client_type=llm_client_type,
        use_reranker=use_reranker,
        reranker_beta=reranker_beta,
        use_decomposer=use_decomposer,
    )


def _build_search_config(base: SearchConfig, overrides: dict[str, Any]) -> SearchConfig:
    """Build a request-scoped ``SearchConfig`` from UI overrides."""
    return SearchConfig(
        top_k=overrides.get("top_k", base.top_k),
        expand_depth=overrides.get("expand_depth", base.expand_depth),
        max_entity_degree=overrides.get("max_entity_degree", base.max_entity_degree),
        max_expansion_per_entity=overrides.get(
            "max_expansion_per_entity", base.max_expansion_per_entity
        ),
        max_expanded_nodes=overrides.get("max_expanded_nodes", base.max_expanded_nodes),
        alpha=overrides.get("alpha", base.alpha),
        depth_scores=base.depth_scores,
        max_results=overrides.get("max_results", base.max_results),
    )


def _results_to_csv(results: list[RetrievalResult]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "rank",
            "chunk_id",
            "article_id",
            "source",
            "depth",
            "vector_score",
            "graph_score",
            "combined_score",
            "text",
        ],
    )
    writer.writeheader()
    for rank, result in enumerate(results, start=1):
        writer.writerow(
            {
                "rank": rank,
                "chunk_id": result.chunk_id,
                "article_id": result.article_id,
                "source": result.source,
                "depth": result.depth,
                "vector_score": f"{result.vector_score:.4f}",
                "graph_score": f"{result.graph_score:.4f}",
                "combined_score": f"{result.combined_score:.4f}",
                "text": result.text,
            }
        )
    return output.getvalue()


def _render_result_card(rank: int, result: RetrievalResult) -> None:
    with st.expander(
        f"#{rank} {result.chunk_id} | {result.source} | score={result.combined_score:.4f}"
    ):
        st.markdown(
            f"""
            **Article:** `{result.article_id}`  
            **Source:** `{result.source}`  
            **Depth:** `{result.depth}`  
            **Vector score:** `{result.vector_score:.4f}`  
            **Graph score:** `{result.graph_score:.4f}`  
            **Combined score:** `{result.combined_score:.4f}`
            """
        )
        st.markdown(f"> {result.text}")


def _render_graph_evidence(graph_repository: Any, results: list[RetrievalResult]) -> None:
    st.subheader("Graph evidence")
    if not results:
        st.write("No results to visualize.")
        return

    top = results[:5]
    entity_counts: dict[str, int] = {}
    for result in top:
        for entity_id in graph_repository.get_chunk_entities(result.chunk_id):
            entity_counts[entity_id] = entity_counts.get(entity_id, 0) + 1

    if not entity_counts:
        st.write("No shared entities found for the top results.")
        return

    sorted_entities = sorted(entity_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    st.write("Top entities mentioned by the top 5 retrieved chunks:")
    for entity_id, count in sorted_entities:
        degree = graph_repository.get_entity_degree(entity_id)
        st.write(f"- `{entity_id}` (mentions={count}, degree={degree})")


def main() -> int:
    st.set_page_config(page_title="PubMed GraphRAG Demo", layout="wide")
    st.title("🧬 PubMed GraphRAG Demo")
    st.markdown(
        "Ask a biomedical question. The demo retrieves semantic chunks from 5,000 "
        "PubMed abstracts using graph-enhanced retrieval."
    )

    with st.sidebar:
        st.header("Model")

        llm_options = ["mock", "openai"]
        if os.environ.get("OLLAMA_URL"):
            llm_options.append("ollama")

        llm_client_type = st.selectbox(
            "LLM client",
            options=llm_options,
            index=0,
            help="Select the LLM used for generation. Ollama only appears if OLLAMA_URL is set.",
        )

        st.header("Retrieval")
        top_k = st.slider("top_k", 1, 50, 10)
        expand_depth = st.slider("expand_depth", 0, 3, 2)
        max_entity_degree = st.slider("max_entity_degree", 10, 2000, 500)
        alpha = st.slider("alpha (vector weight)", 0.0, 1.0, 0.8, step=0.05)
        max_results = st.slider("max_results", 1, 50, 20)

        st.header("Phase 5 options")
        use_decomposer = st.checkbox("Enable query decomposition", value=False)
        use_reranker = st.checkbox("Enable graph re-ranking", value=False)
        reranker_beta = st.slider(
            "reranker beta (original score weight)",
            0.0,
            1.0,
            0.7,
            step=0.05,
            disabled=not use_reranker,
        )

    retrieval_overrides = {
        "top_k": top_k,
        "expand_depth": expand_depth,
        "max_entity_degree": max_entity_degree,
        "alpha": alpha,
        "max_results": max_results,
    }

    try:
        _ensure_artifacts_cached()
        pipeline = _load_pipeline(
            llm_client_type=llm_client_type,
            use_reranker=use_reranker,
            reranker_beta=reranker_beta,
            use_decomposer=use_decomposer,
        )
        base_config = default_search_config()
        search_config = _build_search_config(base_config, retrieval_overrides)
        graph_repository = pipeline.retrieve_documents.graph_expand.graph_repository
    except Exception as exc:
        st.error(f"Failed to load pipeline: {exc}")
        return 1

    query = st.text_input(
        "Question",
        value="What are the risk factors for type 2 diabetes?",
        placeholder="Enter a biomedical question...",
    )

    col1, col2 = st.columns(2)
    retrieve_clicked = col1.button("🔍 Retrieve")
    answer_clicked = col2.button("💬 Answer")

    if retrieve_clicked or answer_clicked:
        with st.spinner("Retrieving..."):
            if use_decomposer:
                sub_queries, results = pipeline.retrieve_decomposed(query, search_config)
                st.write(f"Sub-queries used ({len(sub_queries)}): {sub_queries}")
            else:
                results = pipeline.retrieve_reranked(query, search_config)

        st.subheader(f"Retrieved context ({len(results)} chunks)")
        for rank, result in enumerate(results, start=1):
            _render_result_card(rank, result)

        st.download_button(
            label="Download results as CSV",
            data=_results_to_csv(results),
            file_name="retrieval_results.csv",
            mime="text/csv",
        )

        _render_graph_evidence(graph_repository, results)

        if answer_clicked:
            with st.spinner("Generating answer..."):
                response = pipeline.generate(query, search_config, context=results)
            st.subheader("Answer")
            st.markdown(response.answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
