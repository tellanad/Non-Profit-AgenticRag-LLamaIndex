"""
Tests for the embedder (embedding model + vector store).

NOTE: test_embed_model will download the model (~130MB) on first run.
After that it's cached. The test verifies the model produces
vectors of the expected dimensionality.
"""

import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.embedder import get_embed_model, get_vector_store, get_storage_context


def test_embed_model_loads():
    """Embedding model should load and be ready."""
    model = get_embed_model()
    assert model is not None, "Model failed to load"
    print(f"✓ Embedding model loaded: {model.model_name}")


def test_embed_model_produces_vectors():
    """Model should produce a vector of correct dimensions for bge-small (384)."""
    model = get_embed_model()
    embedding = model.get_text_embedding("This is a test sentence.")

    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) == 384, f"Expected 384 dims, got {len(embedding)}"
    assert all(isinstance(x, float) for x in embedding), "All values should be floats"

    print(f"✓ Embedding produced: {len(embedding)} dimensions")


def test_vector_store_connects():
    """ChromaDB should connect and create/find the collection."""
    store = get_vector_store()
    assert store is not None, "Vector store failed to connect"
    print("✓ ChromaDB vector store connected")


def test_storage_context_creates():
    """StorageContext should wrap the vector store correctly."""
    ctx = get_storage_context()
    assert ctx is not None, "StorageContext failed to create"
    assert ctx.vector_store is not None, "Vector store not attached to context"
    print("✓ StorageContext created with vector store")


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging("DEBUG")

    test_embed_model_loads()
    test_embed_model_produces_vectors()
    test_vector_store_connects()
    test_storage_context_creates()

    print("\n✅ All embedder tests passed!")