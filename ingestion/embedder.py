"""
Embedder — Configures the embedding model and vector store connection.

This module sets up two things:
1. The embedding model (turns text chunks into vectors)
2. The ChromaDB vector store (where vectors are persisted)

LlamaIndex combines embed + store into one step via VectorStoreIndex,
so this module provides the configured components that pipeline.py
will use to build the index.

WHY SEPARATE THIS FROM PIPELINE?
- Embedding model and vector store are infrastructure concerns
- You might swap ChromaDB for Qdrant without changing the pipeline logic
- Testing: you can mock these components independently

TRADEOFF: text-embedding-3-small vs text-embedding-3-large
  small: 1536 dims, $0.02/1M tokens — good for MVP, fast, cheap
  large: 3072 dims, $0.13/1M tokens — better retrieval, 2x storage
  Verdict: Start small, switch to large only if eval shows retrieval gaps.
"""

import chromadb