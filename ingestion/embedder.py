"""
Embedder — Configures the embedding model and vector store connection.

EMBEDDING STRATEGY:
- Using open-source BAAI/bge-small-en-v1.5 (runs locally, free)
- OpenAI is only used for LLM reasoning (gpt-4o), not embeddings
- This saves significant cost: embeddings are called on every chunk
  during ingestion AND every query at runtime

TRADEOFF: bge-small vs bge-large
  bge-small-en-v1.5: 384 dims, ~33M params, fast, low memory
  bge-large-en-v1.5: 1024 dims, ~335M params, better quality, 10x heavier
  Verdict: Start with small. Switch to large if eval shows retrieval gaps.

NOTE: First run will download the model (~130MB). Subsequent runs use cache.
"""

import chromadb

from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)

def get_embed_model() -> HuggingFaceEmbedding:
    """Create and return the local Huggingfaceembedding
     First call downloads the model (~130MB). Cached after that.
     Returns : The HuggingFaceEmbedding  
    """
    
    logger.info(f"Initializing embedding model: {settings.embed_model}")
    
    embed_model= HuggingFaceEmbedding( model_name= settings.embed_model)
    
    return embed_model

def get_vector_store()-> ChromaVectorStore:
    """Create and return the ChromaDB vector store.
    
    Uses persistent storage so vectors survive between runs.
    
     Returns:
        ChromaVectorStore connected to a persistent ChromaDB collection.
    """
    logger.info(
        f"Connecting to ChromaDB: "
        f"path={settings.chroma_persist_dir}, "
        f"collection={settings.chroma_collection_name}"
    )

    db = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = db.get_or_create_collection(settings.chroma_collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    logger.info(
        f"ChromaDB ready: {collection.count()} existing vectors in collection"
    )

    return vector_store

def get_storage_context() -> StorageContext:
    """Create a StorageContext wired to the vector store.

    Returns:
        StorageContext with ChromaDB vector store attached.
    """
    vector_store = get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context
    