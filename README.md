✅ Layer 1: Ingestion Pipeline
   ├── pdf_parser.py    → parse PDF + enrich metadata (TOC sections)
   ├── chunker.py       → split into 512-token nodes
   ├── embedder.py      → local HuggingFace embeddings + ChromaDB
   └── pipeline.py      → orchestrates parse → chunk → embed + store

✅ Layer 2: Query Engines
   ├── vector_engine.py  → similarity search for specific questions
   └── summary_engine.py → full-document summarization

✅ Layer 3: Agent
   ├── tools.py          → wraps engines as agent tools with descriptions
   └── rag_agent.py      → ReActAgent that reasons about which tool to use

✅ Scripts
   ├── ingest.py         → CLI for ingestion
   └── query.py          → interactive chat with the agent


#TODO : FastAPI backend — expose the agent as REST endpoints (your Next.js frontend will call these)

#TODO : Guardrails — input validation, output quality checks

#TODO: Observability — tracing with Arize Phoenix so you can see what's happening

#TODO: Evaluation pipeline — golden test set to measure answer quality

#TODO: Multi-document support — Phase 2 scaling


#_________________________ Optional Based on Metrics :TODO Enterprise _______________________#

#TODO : Model Router / LLM Flexibility

#TODO: Retrieval Quality Improvements
       Embedding Model Swapping
       Hybrid Search (Sparse + Dense)
       Reranker (post-retrieval quality boost)


