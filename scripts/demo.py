#!/usr/bin/env python3
"""Lightweight Streamlit demo for the PubMed GraphRAG pipeline.

Usage:
    streamlit run scripts/demo.py

The demo loads the Phase 1/2 artifacts once, then answers biomedical
questions using the graph-enhanced retriever.  Optional query decomposition
and graph re-ranking can be enabled from the sidebar.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on the path when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, RetrievalConfig
from src.graph_reranker import create_graph_reranker
from src.llm_client import create_llm_client
from src.query_decomposer import QueryDecomposer, DecomposerConfig
from src.rag_pipeline import RAGPipeline, create_rag_pipeline
from src.retriever import RetrievalResult, create_retriever

try:
    import streamlit as st
except ImportError as exc:
    print(
        "Streamlit is not installed. Install it with: pip install streamlit\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _build_retrieval_config(base: RetrievalConfig, overrides: dict[str, Any]) -> RetrievalConfig:
    """Rebuild a RetrievalConfig from user-provided overrides."""
    return RetrievalConfig(
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


@st.cache_resource(show_spinner="Loading PubMed GraphRAG index...")
def _load_pipeline(
    retrieval_overrides: dict[str, Any],
    llm_client_type: str,
    use_decomposer: bool,
    use_reranker: bool,
    reranker_beta: float,
    use_pagerank: bool,
) -> RAGPipeline:
    """Load the retriever, optional reranker, and LLM once per session."""
    base_config = AppConfig.default()
    retrieval_config = _build_retrieval_config(base_config.retrieval, retrieval_overrides)
    config = AppConfig(
        neo4j=base_config.neo4j,
        embedding=base_config.embedding,
        artifact=base_config.artifact,
        retrieval=retrieval_config,
        rerank=base_config.rerank,
        decomposer=base_config.decomposer,
    )

    llm = create_llm_client(llm_client_type)
    retriever = create_retriever(config)

    decomposer: QueryDecomposer | None = None
    if use_decomposer:
        decomposer = QueryDecomposer(llm=llm, config=DecomposerConfig(enabled=True))

    reranker = None
    if use_reranker:
        reranker = create_graph_reranker(
            index=retriever.index,
            enabled=True,
            beta=reranker_beta,
            use_pagerank=use_pagerank,
            app_config=config,
        )

    return create_rag_pipeline(
        config=config,
        llm=llm,
        decomposer=decomposer,
        reranker=reranker,
    )


def _results_to_csv(results: list[RetrievalResult]) -> str:
    """Render retrieval results as CSV text."""
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
    """Render one retrieval result as an expander card."""
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


def _render_graph_evidence(pipeline: RAGPipeline, results: list[RetrievalResult]) -> None:
    """Show a simple text summary of entity connections for the top results."""
    st.subheader("Graph evidence")
    if not results:
        st.write("No results to visualize.")
        return

    top = results[:5]
    entity_counts: dict[str, int] = {}
    for result in top:
        for entity_id in pipeline.retriever.index.chunk_entities.get(result.chunk_id, set()):
            entity_counts[entity_id] = entity_counts.get(entity_id, 0) + 1

    if not entity_counts:
        st.write("No shared entities found for the top results.")
        return

    sorted_entities = sorted(entity_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    st.write("Top entities mentioned by the top 5 retrieved chunks:")
    for entity_id, count in sorted_entities:
        degree = pipeline.retriever.index.entity_degrees.get(entity_id, 0)
        st.write(f"- `{entity_id}` (mentions={count}, degree={degree})")


def main() -> int:
    st.set_page_config(page_title="PubMed GraphRAG Demo", layout="wide")
    st.title("🧬 PubMed GraphRAG Demo")
    st.markdown(
        "Ask a biomedical question. The demo retrieves semantic chunks from 5,000 "
        "PubMed abstracts and optionally boosts the ranking with graph signals."
    )

    with st.sidebar:
        st.header("Model")
        llm_client_type = st.selectbox(
            "LLM client",
            options=["mock", "openai", "ollama"],
            index=0,
            help="Select the LLM used for generation and query decomposition.",
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
        use_pagerank = st.checkbox(
            "Use Neo4j GDS PageRank",
            value=False,
            disabled=not use_reranker,
            help="Requires a running Neo4j instance with a GDS graph named 'pubmed-graph'.",
        )

    retrieval_overrides = {
        "top_k": top_k,
        "expand_depth": expand_depth,
        "max_entity_degree": max_entity_degree,
        "alpha": alpha,
        "max_results": max_results,
    }

    try:
        pipeline = _load_pipeline(
            retrieval_overrides=retrieval_overrides,
            llm_client_type=llm_client_type,
            use_decomposer=use_decomposer,
            use_reranker=use_reranker,
            reranker_beta=reranker_beta,
            use_pagerank=use_pagerank,
        )
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
                sub_queries, results = pipeline.retrieve_decomposed(query)
                st.write(f"Sub-queries used ({len(sub_queries)}): {sub_queries}")
            else:
                results = pipeline.retrieve_reranked(query)

        st.subheader(f"Retrieved context ({len(results)} chunks)")
        for rank, result in enumerate(results, start=1):
            _render_result_card(rank, result)

        st.download_button(
            label="Download results as CSV",
            data=_results_to_csv(results),
            file_name="retrieval_results.csv",
            mime="text/csv",
        )

        _render_graph_evidence(pipeline, results)

        if answer_clicked:
            with st.spinner("Generating answer..."):
                response = pipeline.generate(query, context=results)
            st.subheader("Answer")
            st.markdown(response.answer)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
