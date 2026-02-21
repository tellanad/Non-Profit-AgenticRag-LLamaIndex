"""
Vector Query Engine — For specific, targeted questions.

This engine does similarity search against your ChromaDB embeddings
to find the most relevant chunks, then uses the LLM to synthesize
an answer from those chunks.

USE CASES:
- "What is the refund policy?"
- "How much was the marketing budget in Q3?"
- "Who is on the board of directors?"

HOW IT WORKS:
1. User query gets embedded using the same HuggingFace model
2. ChromaDB finds the closest matching chunks (similarity_top_k)
3. Those chunks are passed to GPT-4o-mini as context
4. GPT-4o-mini synthesizes a natural language answer

KEY DISTINCTION:
- Ingestion: created the index and stored vectors (write operation)
- This engine: loads the existing index and searches it (read operation)
- No re-embedding of documents happens here
"""

from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

from ingestion.embedder import get_embed_model, get_vector_store
from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def get_vector_query_engine():
    """Load the existing index from ChromaDB and return a query engine.

    This connects to the ChromaDB collection that was populated
    during ingestion, loads the index, and wraps it as a query engine
    with the configured LLM for answer synthesis.

    Returns:
        QueryEngine configured for similarity search + LLM synthesis.
    """
    # ── Load the embedding model (same one used during ingestion) ───
    embed_model = get_embed_model()

    # ── Connect to existing ChromaDB store ─────────────────────────
    vector_store = get_vector_store()

    # ── Load the index from the existing vector store ──────────────
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # ── Create the LLM for answer synthesis ────────────────────────
    llm = OpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    # ── Build the query engine ─────────────────────────────────────
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=settings.similarity_top_k,
        response_mode=settings.response_mode,
    )

    logger.info(
        f"Vector query engine ready: "
        f"top_k={settings.similarity_top_k}, "
        f"response_mode={settings.response_mode}, "
        f"llm={settings.llm_model}"
    )

    return query_engine
