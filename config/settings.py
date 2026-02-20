"""
Centralized configuration using Pydantic Settings.

WHY PYDANTIC SETTINGS?
- Type-safe: catches misconfigurations at startup, not mid-query
- Env-aware: automatically loads from .env file
- Validated: e.g., chunk_size must be > 0
- Documented: each field has a description
- Testable: override any setting in tests via constructor

TRADEOFF vs plain os.environ:
  + Type safety, validation, defaults, IDE autocomplete
  - Extra dependency (pydantic-settings)
  Verdict: Always worth it for anything beyond a toy project
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ── LLM Configuration ──────────────────────────────────────────────
    openai_api_key: str = Field(
        ...,
        description="OpenAI API key for LLM and embedding calls",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="LLM model name. Options: gpt-4o, gpt-4o-mini, gpt-3.5-turbo",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature. 0 = deterministic (best for RAG), higher = creative",
    )

    # ── Embedding Configuration ────────────────────────────────────────
    embed_model: str = Field(
        default="tBAAI/bge-small-en-v1.5",
        description=(
            "Embedding model. Options:\n"
            "  text-embedding-3-small: $0.02/1M tokens, 1536 dims (MVP)\n"
            "  text-embedding-3-large: $0.13/1M tokens, 3072 dims (better quality)\n"
            "  local:BAAI/bge-large-en-v1.5: Free, 1024 dims (privacy-first)"
        ),
    )

    # ── Chunking Configuration ─────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        gt=0,
        description=(
            "Tokens per chunk. Tradeoffs:\n"
            "  256-512: More precise retrieval, more chunks to search\n"
            "  512-1024: Balanced (recommended start)\n"
            "  1024-2048: More context per chunk, may dilute relevance"
        ),
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Token overlap between chunks. ~10% of chunk_size is a good default.",
    )

    # ── Retrieval Configuration ────────────────────────────────────────
    similarity_top_k: int = Field(
        default=5,
        gt=0,
        description=(
            "Number of chunks to retrieve. Tradeoffs:\n"
            "  3-5: Fast, precise, lower cost\n"
            "  5-10: Better recall, use with reranking\n"
            "  10+: Only if using a reranker to filter down"
        ),
    )
    response_mode: str = Field(
        default="compact",
        description=(
            "How to synthesize the final answer. Options:\n"
            "  compact: Stuffs chunks into one prompt (balanced)\n"
            "  refine: Iteratively refines (detailed but costly)\n"
            "  tree_summarize: Bottom-up tree (good for summaries)\n"
            "  simple_summarize: Truncates to fit (fast but lossy)"
        ),
    )

    # ── Agent Configuration ────────────────────────────────────────────
    max_agent_iterations: int = Field(
        default=6,
        gt=0,
        le=20,
        description=(
            "Max reasoning loops for the agent. Prevents infinite loops.\n"
            "  3-4: Fast, may miss complex queries\n"
            "  6: Good default\n"
            "  10+: For very complex multi-step reasoning"
        ),
    )
    agent_system_prompt: str = Field(
        default=(
            "You are a helpful assistant for a non-profit organization. "
            "You answer questions about organizational documents. "
            "Always cite which part of the document your answer comes from. "
            "If you cannot find the answer, say so clearly — do not make things up."
        ),
        description="System prompt that guides the agent's behavior.",
    )

    # ── Storage Configuration ──────────────────────────────────────────
    chroma_persist_dir: str = Field(
        default="./storage/chroma_db",
        description="Directory for ChromaDB persistent storage.",
    )
    chroma_collection_name: str = Field(
        default="nonprofit_docs",
        description="ChromaDB collection name.",
    )

    # ── Observability ──────────────────────────────────────────────────
    enable_tracing: bool = Field(
        default=True,
        description="Enable Arize Phoenix tracing for observability.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",  # ignore unknown env vars
    }
