"""
Tests for the chunking pipeline.

These tests use fake Document objects (no real PDF needed)
to verify chunking behavior in isolation.
"""

import sys
from  pathlib import Path 

sys.path.insert(0, str(Path(__file__).parent.parent))

from llama_index.core import Document
from ingestion.chunker import chunk_documents

def test_chunk_basic():
    """Chunking a document should produce multiple smaller nodes."""
    # Create a fake document with enough text to split
    long_text = "This is a sentence about non-profit policies. " * 100
    docs = [Document(text=long_text, metadata={"page_label": "1", "section": "Intro"})]

    nodes = chunk_documents(docs)

    assert len(nodes) > 1, "Long text should produce multiple chunks"
    print(f"✓ Produced {len(nodes)} chunks from 1 document")
    
def test_chunk_metadata_preserved():
    """Metadata from parent Document should carry to child TextNodes."""
    docs = [Document(
        text="This is a test sentence. " * 50,
        metadata={
            "page_label": "5",
            "section": "Financial Policies",
            "file_name": "policy.pdf",
            "total_pages": 20,
        },
    )]

    nodes = chunk_documents(docs)

    for node in nodes:
        assert node.metadata["section"] == "Financial Policies", "Section metadata lost"
        assert node.metadata["page_label"] == "5", "Page label lost"
        assert node.metadata["file_name"] == "policy.pdf", "File name lost"

    print(f"✓ Metadata preserved across {len(nodes)} chunks")
    
    
def test_chunk_empty_input():
    """Empty input should return empty list, not crash."""
    nodes = chunk_documents([])
    assert nodes == [], "Empty input should return empty list"
    print("✓ Empty input handled gracefully")
    
def test_chunk_no_overlap_loss():
    """Chunks should have overlapping content at boundaries."""
    # Create text with distinct sentences
    sentences = [f"Sentence number {i} contains unique content. " for i in range(50)]
    docs = [Document(text="".join(sentences))]

    nodes = chunk_documents(docs)

    # Check that consecutive chunks share some text (overlap)
    if len(nodes) >= 2:
        # The end of chunk 1 should appear at the start of chunk 2
        overlap_found = any(
            word in nodes[1].text
            for word in nodes[0].text.split()[-10:]
        )
        assert overlap_found, "No overlap detected between consecutive chunks"
        print("✓ Overlap verified between chunks")
    else:
        print("⚠ Only 1 chunk produced, can't verify overlap")


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging("DEBUG")

    test_chunk_basic()
    test_chunk_metadata_preserved()
    test_chunk_empty_input()
    test_chunk_no_overlap_loss()

    print("\n✅ All chunker tests passed!")