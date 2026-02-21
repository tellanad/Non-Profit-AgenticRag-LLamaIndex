"""
Summary Query Engine — For broad, overview questions.

This engine reads ALL nodes from the document and synthesizes
a comprehensive summary using tree_summarize mode.

USE CASES:
- "What is this document about?"
- "Summarize the key points"
- "Give me an overview of the organization's programs"

HOW IT WORKS:
1. Loads ALL nodes from ChromaDB (not just top-k matches)
2. Builds a SummaryIndex over all nodes
3. Uses tree_summarize: splits nodes into groups, summarizes each
   group, then summarizes the summaries (bottom-up tree)
4. LLM (OpenAI) generates the final synthesized summary

TRADEOFF vs Vector Engine:
  Vector: Precise, fast, cheap (only processes top-k chunks)
  Summary: Comprehensive, slower, costlier (processes ALL chunks)
  The agent decides which to use based on the question.

WHY tree_summarize?
- "compact" would try to stuff all nodes into one prompt — fails for large docs
- "refine" processes chunks sequentially — slow, order-dependent
- "tree_summarize" is hierarchical — handles any document size, parallelizable
"""

from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.llms.openai import OpenAI

from ingestion.embedder import get_embed_model, get_vector_store
from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def get_summary_query_engine():
    """Build a summary query engine over all document nodes.

    Loads the existing vectors from ChromaDB, extracts all nodes,
    and builds a SummaryIndex for comprehensive summarization.

    Returns:
        QueryEngine configured for full-document summarization.
    """
    # ── Load index from ChromaDB to access all nodes ───────────────
    embed_model = get_embed_model()
    vector_store = get_vector_store()

    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # ── Extract all nodes from the index ───────────────────────────
    retriever = vector_index.as_retriever(
        similarity_top_k=1000,
    )
    all_nodes = retriever.retrieve("document content summary overview")
    nodes = [node_with_score.node for node_with_score in all_nodes]

    if not nodes:
        logger.warning("No nodes found in ChromaDB — run ingestion first")
        raise ValueError("No nodes in vector store. Run ingestion pipeline first.")

    logger.info(f"Loaded {len(nodes)} nodes for summary index")

    # ── Build SummaryIndex ─────────────────────────────────────────
    summary_index = SummaryIndex(nodes=nodes, embed_model=embed_model)

    # ── Create the LLM ─────────────────────────────────────────────
    llm = OpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    # ── Build the query engine ─────────────────────────────────────
    query_engine = summary_index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )

    logger.info(
        f"Summary query engine ready: "
        f"{len(nodes)} nodes, "
        f"response_mode=tree_summarize, "
        f"llm={settings.llm_model}"
    )

    return query_engine
