"""
Integration test for the full ingestion pipeline.

This is an end-to-end test — it runs the real pipeline against
a real PDF, producing real embeddings stored in a test ChromaDB.

NOTE: This test:
- Requires a sample PDF in data/documents/sample.pdf
- Will download the embedding model on first run (~130MB)
- Creates vectors in your ChromaDB storage
- Takes 10-30 seconds depending on PDF size
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from ingestion.pipeline import run_ingestion


def test_full_pipeline():
    """End-to-end: ingest a real PDF and verify the index works."""
    pdf_path = Path("data/documents/sample.pdf")

    if not pdf_path.exists():
        print(f"⚠ Skipping: {pdf_path} not found")
        return

    # Run the full pipeline
    index = run_ingestion(pdf_path)

    # Verify: index should exist and be queryable
    assert index is not None, "Pipeline returned None"

    # Verify: try a simple query against the index
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("What is this document about?")

    assert len(nodes) > 0, "Retriever returned no nodes"
    print(f"✓ Pipeline complete — index created with {len(nodes)} retrievable nodes")
    print(f"✓ First result: {nodes[0].node.text[:200]}...")


def test_pipeline_returns_queryable_index():
    """Verify the returned index can retrieve nodes with metadata."""
    pdf_path = Path("data/documents/sample.pdf")

    if not pdf_path.exists():
        print(f"⚠ Skipping: {pdf_path} not found")
        return

    index = run_ingestion(pdf_path)

    # Use retriever to get raw nodes (not synthesized answer)
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve("What is this document about?")

    assert len(nodes) > 0
    print(f"✓ Retrieved {len(nodes)} nodes")
    for node in nodes:
        print(f"  Section: {node.node.metadata.get('section')}")
        print(f"  Text: {node.node.text[:150]}...")

    # Check that metadata survived the full pipeline
    for node in nodes:
        metadata = node.node.metadata
        assert "file_name" in metadata, "file_name missing from metadata"
        assert "section" in metadata, "section missing from metadata"
        print(f"  ✓ Node from section: '{metadata.get('section')}', "
              f"page: {metadata.get('page_label')}")

    print(f"✓ Retrieved {len(nodes)} nodes with intact metadata")


if __name__ == "__main__":
    setup_logging("DEBUG")

    test_full_pipeline()
    print()
    test_pipeline_returns_queryable_index()

    print("\n✅ All pipeline tests passed!")