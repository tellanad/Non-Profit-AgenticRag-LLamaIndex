"""
Ingestion Pipeline — Orchestrates the full flow: parse → chunk → embed + store.

This is the single entry point for ingesting a PDF into the vector store.
scripts/ingest.py calls this function.

WHY A SEPARATE PIPELINE FILE?
- Parser, chunker, and embedder are independent units (testable alone)
- Pipeline wires them together and handles the orchestration concerns:
  timing, logging, error handling
- When you add batch processing (Phase 2), you change this file only
"""


import time
from pathlib import Path

from llama_index.core import VectorStoreIndex

from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import get_embed_model, get_storage_context

from config.logging_config import get_logger

logger = get_logger(__name__)

def run_ingestion(file_path:str | Path) ->VectorStoreIndex:
    """Run the full ingestion pipeline for a single PDF.

    Flow: parse PDF → chunk into nodes → embed + store in ChromaDB

    Args:
        file_path: Path to the PDF file.

    Returns:
        VectorStoreIndex ready for querying.
    """
    start_time = time.time()
    file_path = Path(file_path)
    logger.info(f"Starting ingestion pipeline for: {file_path.name}")
    # ── Step 1: Parse ───────────────────────────────────────────────
    documents = parse_pdf(file_path)
    logger.info(f"Step 1 complete: {len(documents)} pages parsed")
    # ── Step 2: Chunk ───────────────────────────────────────────────
    nodes = chunk_documents(documents)
    logger.info(f"Step 2 complete: {len(nodes)} chunks created")
    # ── Step 3: Embed + Store ───────────────────────────────────────
    embed_model = get_embed_model()
    storage_context = get_storage_context()
    
    # This single call does two things:
    # 1. Embeds every node using the embed model
    # 2. Stores the vectors in ChromaDB via the storage context
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    
    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    logger.info(
        f"Ingestion complete: {file_path.name} → "
        f"{len(nodes)} chunks indexed in {elapsed:.1f}s"
    )
    
    #TODO: Phase 2 — Add duplicate detection before inserting
    # Check if file_name already exists in the vector store metadata
    # to avoid re-indexing the same document

    return index
