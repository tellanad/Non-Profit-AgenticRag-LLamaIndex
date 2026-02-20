# Enterprise Agentic RAG with LlamaIndex — Architecture Guide

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Layers](#2-architecture-layers)
3. [Phase 1: MVP — Single Document Agentic RAG](#3-phase-1-mvp)
4. [Phase 2: Multi-Document Scaling](#4-phase-2-multi-document)
5. [Project Structure](#5-project-structure)
6. [Component Deep-Dives with Tradeoffs](#6-component-deep-dives)
7. [Enterprise Patterns](#7-enterprise-patterns)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. System Overview

### What We're Building

An **Agentic RAG** system goes beyond naive "retrieve-then-generate." Instead of a fixed pipeline, an *agent* decides **when** to retrieve, **what** to retrieve, and **whether the answer is good enough** — looping back if it isn't. Think of it as giving the LLM a brain that can use tools (search, summarize, calculate) rather than just reading chunks.

```
User Query
    │
    ▼
┌──────────────┐
│   Agent       │  ← Decides what to do (reason, retrieve, summarize, ask follow-up)
│  (AgentRunner)│
└──────┬───────┘
       │ picks a tool
       ▼
┌──────────────────────────────────┐
│         Tool Layer               │
│  ┌──────────┐  ┌──────────────┐  │
│  │ Vector   │  │ Summary      │  │
│  │ Query    │  │ Query        │  │
│  │ Engine   │  │ Engine       │  │
│  └──────────┘  └──────────────┘  │
│  ┌──────────┐  ┌──────────────┐  │
│  │ Custom   │  │ Metadata     │  │
│  │ Functions│  │ Filters      │  │
│  └──────────┘  └──────────────┘  │
└──────────────────────────────────┘
       │
       ▼
┌──────────────┐
│  Vector Store │  ← ChromaDB (MVP) → Qdrant/Pinecone (Scale)
└──────────────┘
       │
       ▼
┌──────────────┐
│  PDF Document │
└──────────────┘
```

### Why Agentic, Not Just RAG?

| Approach | How It Works | Limitation |
|----------|-------------|------------|
| **Naive RAG** | Query → Retrieve top-k → Generate | Can't handle multi-step reasoning, no self-correction |
| **Advanced RAG** | Query rewriting + reranking + hybrid search | Still a fixed pipeline, no decision-making |
| **Agentic RAG** | Agent reasons about what tools to use, loops until satisfied | More complex, higher latency, but far more capable |

**Tradeoff**: Agentic RAG adds ~1-3 extra LLM calls per query (latency + cost), but dramatically improves answer quality for complex questions. For simple lookups, it gracefully falls back to single-step retrieval.

---

## 2. Architecture Layers

We use a **3-layer architecture** that separates concerns cleanly:

### Layer 1: Ingestion Pipeline
Responsible for: parsing PDFs, chunking text, generating embeddings, storing in vector DB.

### Layer 2: Retrieval & Query Engines
Responsible for: vector search, summary generation, metadata filtering. Each becomes a "tool" the agent can use.

### Layer 3: Agent Orchestration
Responsible for: reasoning about which tools to call, synthesizing answers, self-correcting.

```
┌─────────────────────────────────────────────────────┐
│                 Layer 3: AGENT                       │
│  AgentRunner + ReAct reasoning + conversation memory │
├─────────────────────────────────────────────────────┤
│                 Layer 2: TOOLS                       │
│  VectorQueryEngine │ SummaryQueryEngine │ Functions  │
├─────────────────────────────────────────────────────┤
│                 Layer 1: INGESTION                   │
│  PDF Parser → Chunker → Embedder → Vector Store     │
└─────────────────────────────────────────────────────┘
```

**Why 3 layers?** Each layer can be tested, swapped, and scaled independently. You can change your vector DB without touching the agent. You can add new tools without changing ingestion.

---

## 3. Phase 1: MVP — Single Document Agentic RAG

### Goal
Take one PDF document, ingest it, and build an agent that can answer questions about it using both semantic search AND summarization.

### 3.1 Ingestion Pipeline

#### PDF Parsing

**Option A: LlamaParse (Recommended for complex PDFs)**
- Handles tables, headers, footnotes, multi-column layouts
- Cloud-based service (free tier: 1000 pages/day)
- Tradeoff: adds external dependency + network latency

**Option B: PyMuPDF / pdfplumber (Self-hosted)**
- No external dependency, runs locally
- Struggles with complex layouts, tables
- Tradeoff: free and private, but lower quality parsing

**Recommendation for MVP**: Start with PyMuPDF for simplicity. Switch to LlamaParse when you hit parsing quality issues.

#### Chunking Strategy

```python
# MVP: SentenceSplitter (simple, effective)
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,      # tokens per chunk
    chunk_overlap=50,    # overlap for context continuity
)
```

**Tradeoff Table: Chunking Approaches**

| Strategy | Chunk Size | Best For | Weakness |
|----------|-----------|----------|----------|
| `SentenceSplitter` (512 tokens) | Fixed | MVP, general docs | Ignores document structure |
| `SemanticSplitter` | Dynamic | Narrative content | Slower, needs embedding model |
| `HierarchicalNodeParser` | Multi-level | Structured docs with sections | More complex to set up |
| `SentenceWindowNodeParser` | Sentence + window | Precise retrieval with context | Higher storage, more complex |

**MVP Choice**: `SentenceSplitter` with 512 tokens, 50 token overlap. It's the 80/20 choice — good enough for most documents, simple to understand and debug.

#### Embedding Model

```python
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",  # $0.02/1M tokens
    # model="text-embedding-3-large",  # $0.13/1M tokens, better quality
)
```

**Tradeoff: Embedding Model Choice**

| Model | Dimensions | Cost | Quality |
|-------|-----------|------|---------|
| `text-embedding-3-small` | 1536 | $0.02/1M tokens | Good for MVP |
| `text-embedding-3-large` | 3072 | $0.13/1M tokens | Better retrieval, higher storage |
| `BAAI/bge-large-en-v1.5` (local) | 1024 | Free (compute only) | Good, fully private |

**MVP Choice**: `text-embedding-3-small`. Cheapest, good enough, and you can switch later without rebuilding (just re-index).

#### Vector Store

```python
# MVP: ChromaDB (embedded, zero setup)
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./storage/chroma_db")
chroma_collection = db.get_or_create_collection("nonprofit_docs")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
```

**Tradeoff: Vector Store Choice**

| Store | Type | Best For | Limitation |
|-------|------|----------|------------|
| **ChromaDB** | Embedded | MVP, prototyping | No production clustering |
| **Qdrant** | Self-hosted or cloud | Production, filtering | More ops overhead |
| **Pinecone** | Managed cloud | Scale without ops | Vendor lock-in, cost |
| **pgvector** | Postgres extension | Teams already on Postgres | Lower performance at scale |

**MVP Choice**: ChromaDB with persistent storage. Zero setup, good enough for single-doc. Migrate to Qdrant when you need metadata filtering at scale.

### 3.2 Query Engines (The Agent's Tools)

The key insight: we don't give the agent raw chunks. We give it **tools** — each tool is a query engine wrapped with a name and description so the agent knows when to use it.

#### Tool 1: Vector Query Engine (for specific questions)
```python
from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex.from_vector_store(vector_store)
vector_query_engine = vector_index.as_query_engine(
    similarity_top_k=5,         # retrieve top 5 chunks
    response_mode="compact",    # synthesize a concise answer
)
```

#### Tool 2: Summary Query Engine (for broad questions)
```python
from llama_index.core import SummaryIndex

summary_index = SummaryIndex(nodes)  # uses ALL nodes
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",  # hierarchical summarization
)
```

#### Wrapping as Agent Tools
```python
from llama_index.core.tools import QueryEngineTool

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_search",
    description=(
        "Useful for finding specific facts, policies, numbers, or details "
        "within the document. Use when the question asks about a specific topic."
    ),
)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="document_summary",
    description=(
        "Useful for getting a high-level overview or summary of the entire document. "
        "Use when the question is broad or asks 'what is this document about'."
    ),
)
```

**Why two tools?** Vector search is great for "What is the refund policy?" but terrible for "Summarize this document." The agent learns to pick the right tool based on the question. This is the core of agentic RAG.

### 3.3 Agent Setup

```python
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o", temperature=0)

agent = ReActAgent.from_tools(
    tools=[vector_tool, summary_tool],
    llm=llm,
    verbose=True,          # see the agent's reasoning (essential for debugging)
    max_iterations=6,      # prevent infinite loops
    system_prompt=(
        "You are a helpful assistant for a non-profit organization. "
        "You answer questions about organizational documents. "
        "Always cite which part of the document your answer comes from. "
        "If you cannot find the answer, say so clearly."
    ),
)

# Usage
response = agent.chat("What are the key programs mentioned in this document?")
```

**Tradeoff: Agent Type**

| Agent | Reasoning | When to Use |
|-------|-----------|-------------|
| **ReActAgent** | Think → Act → Observe loop | MVP, transparent reasoning, good for debugging |
| **OpenAIAgent** | Uses OpenAI function calling | Faster (fewer tokens), but less transparent |
| **FunctionCallingAgent** | Generic function calling | Multi-provider support |

**MVP Choice**: `ReActAgent`. Its reasoning is verbose and transparent — you can see exactly why it picked a tool. Essential for learning and debugging.

### 3.4 MVP File Structure

```
AgenticRag_LlamaIndex/
├── config/
│   ├── __init__.py
│   └── settings.py          # All configuration (API keys, model names, chunk sizes)
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py         # PDF loading and parsing
│   ├── chunker.py            # Chunking strategies
│   └── embedder.py           # Embedding pipeline
├── engines/
│   ├── __init__.py
│   ├── vector_engine.py      # Vector query engine setup
│   └── summary_engine.py     # Summary query engine setup
├── agent/
│   ├── __init__.py
│   ├── tools.py              # Tool definitions
│   └── rag_agent.py          # Agent creation and configuration
├── storage/
│   └── chroma_db/            # Persistent vector store (gitignored)
├── data/
│   └── documents/            # Source PDFs
├── scripts/
│   ├── ingest.py             # Run ingestion pipeline
│   └── query.py              # Interactive query interface
├── tests/
│   ├── test_ingestion.py
│   ├── test_engines.py
│   └── test_agent.py
├── .env                      # API keys (gitignored)
├── .env.example              # Template for env vars
├── requirements.txt
├── pyproject.toml
└── README.md
```

**Why this structure?** Each folder maps to an architecture layer. You can test ingestion without the agent. You can swap the vector engine without touching the summary engine. When you scale to multi-doc, you add to this structure rather than rewriting it.

---

## 4. Phase 2: Multi-Document Scaling

### The Multi-Document Agent Pattern

When you go from 1 document to N documents, the architecture evolves into a **hierarchical agent system**:

```
                    ┌─────────────────────┐
                    │   Top-Level Agent    │
                    │ (Routes to doc agents│
                    │  or decomposes query)│
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌──────────────┐ ┌──────────────┐  ┌──────────────┐
     │  Doc Agent 1  │ │  Doc Agent 2  │  │  Doc Agent N  │
     │  (Policy.pdf) │ │  (Report.pdf) │  │  (Manual.pdf) │
     │               │ │               │  │               │
     │ • Vector tool │ │ • Vector tool │  │ • Vector tool │
     │ • Summary tool│ │ • Summary tool│  │ • Summary tool│
     └──────────────┘ └──────────────┘  └──────────────┘
```

### How It Works

1. **Each document gets its own agent** with vector + summary tools (exactly like your MVP, but replicated)
2. **A top-level agent** receives the user query and decides:
   - Which document agent(s) to consult
   - Whether to decompose the query into sub-questions (via `SubQuestionQueryEngine`)
   - How to synthesize answers from multiple sources

### Key Code Pattern

```python
from llama_index.core.tools import QueryEngineTool

# Each document becomes a tool for the top-level agent
doc_tools = []
for doc_name, doc_agent in document_agents.items():
    tool = QueryEngineTool.from_defaults(
        query_engine=doc_agent,
        name=f"agent_{doc_name}",
        description=f"Answers questions about {doc_name}. "
                    f"Contains information about {doc_metadata[doc_name]['summary']}",
    )
    doc_tools.append(tool)

# Top-level agent routes queries to document agents
top_agent = ReActAgent.from_tools(
    tools=doc_tools,
    llm=llm,
    verbose=True,
    system_prompt="You are the main assistant. Route questions to the appropriate document agent.",
)
```

### Scaling Tradeoffs

| Concern | MVP (1 doc) | Scale (N docs) | Tradeoff |
|---------|-------------|-----------------|----------|
| **Index** | Single VectorStoreIndex | Index per document OR single merged index | Per-doc = better isolation, merged = simpler but noisy |
| **Routing** | Not needed | Top-level agent routes by description | Good descriptions are critical for routing accuracy |
| **Cost** | ~2-3 LLM calls/query | ~3-5+ LLM calls/query | More docs = more reasoning steps = higher cost |
| **Latency** | ~2-4 seconds | ~5-15 seconds | Parallel tool calls help, but there's a floor |
| **Vector DB** | ChromaDB | Qdrant with namespaces/collections | Need metadata filtering, multi-tenancy |

### When to Use SubQuestionQueryEngine

Use it when queries span multiple documents: "Compare the budget allocations in the 2023 and 2024 annual reports."

```python
from llama_index.core.query_engine import SubQuestionQueryEngine

sub_question_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=doc_tools,
    llm=llm,
)
# Automatically decomposes "Compare X in doc A vs doc B"
# into sub-questions routed to each doc agent
```

**Tradeoff**: Adds 1-2 extra LLM calls for decomposition, but handles cross-document queries that a single retrieval cannot.

---

## 5. Project Structure (Full Scale)

```
AgenticRag_LlamaIndex/
├── config/
│   ├── __init__.py
│   ├── settings.py               # Pydantic settings with env var loading
│   └── logging_config.py         # Structured logging setup
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py             # PDF parsing (PyMuPDF / LlamaParse)
│   ├── chunker.py                # Pluggable chunking strategies
│   ├── embedder.py               # Embedding pipeline
│   └── pipeline.py               # Orchestrates: parse → chunk → embed → store
├── engines/
│   ├── __init__.py
│   ├── vector_engine.py          # Vector query engine factory
│   ├── summary_engine.py         # Summary query engine factory
│   └── engine_factory.py         # Creates engines per document
├── agent/
│   ├── __init__.py
│   ├── tools.py                  # Tool definitions and registry
│   ├── document_agent.py         # Per-document agent (Phase 2)
│   ├── top_agent.py              # Top-level routing agent (Phase 2)
│   └── rag_agent.py              # Single-doc agent (Phase 1)
├── guardrails/
│   ├── __init__.py
│   ├── input_validator.py        # Query validation, PII detection
│   └── output_validator.py       # Response quality checks, hallucination detection
├── observability/
│   ├── __init__.py
│   ├── tracing.py                # OpenTelemetry / Phoenix setup
│   └── metrics.py                # Custom metrics (latency, cost, quality)
├── evaluation/
│   ├── __init__.py
│   ├── test_cases.json           # Golden Q&A pairs
│   ├── evaluator.py              # Automated eval pipeline
│   └── reports/                  # Eval results
├── storage/
│   ├── chroma_db/                # MVP vector store
│   └── index_store/              # Persisted index metadata
├── data/
│   └── documents/                # Source PDFs
├── scripts/
│   ├── ingest.py                 # Run ingestion
│   ├── query.py                  # Interactive query
│   ├── evaluate.py               # Run evaluations
│   └── serve.py                  # FastAPI server (Phase 2)
├── api/                          # Phase 2: REST API
│   ├── __init__.py
│   ├── main.py                   # FastAPI app
│   ├── routes.py                 # API endpoints
│   └── schemas.py                # Request/response models
├── tests/
│   ├── test_ingestion.py
│   ├── test_engines.py
│   ├── test_agent.py
│   └── test_api.py
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 6. Component Deep-Dives with Tradeoffs

### 6.1 Configuration Management

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    openai_api_key: str
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # Embedding
    embed_model: str = "text-embedding-3-small"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    similarity_top_k: int = 5

    # Agent
    max_agent_iterations: int = 6

    # Vector Store
    chroma_persist_dir: str = "./storage/chroma_db"

    class Config:
        env_file = ".env"
```

**Why Pydantic Settings?** Type-safe, validates at startup, loads from `.env` automatically. When something's misconfigured, you know immediately — not 5 minutes into a query.

### 6.2 Response Modes

LlamaIndex offers several response synthesis modes. This choice significantly affects answer quality:

| Mode | How It Works | Best For | Cost |
|------|-------------|----------|------|
| `refine` | Iteratively refines answer with each chunk | Detailed answers | High (1 LLM call per chunk) |
| `compact` | Stuffs max chunks into one prompt, then refines | Balanced quality/cost | Medium |
| `tree_summarize` | Builds answer tree bottom-up | Long documents, summaries | Medium-High |
| `simple_summarize` | Truncates to fit one prompt | Quick answers | Low |
| `no_text` | Returns retrieved nodes only | When you need raw chunks | Zero LLM cost |

**MVP Choice**: `compact` for vector queries (good balance), `tree_summarize` for summaries (handles full documents well).

### 6.3 Retrieval Enhancements (Post-MVP)

These are improvements to add incrementally after your MVP works:

**Hybrid Search** (add keyword search alongside vector)
```python
# Combines BM25 (keyword) + vector similarity
# Catches exact matches that embeddings miss
# Tradeoff: slight setup complexity, but significant recall improvement
```

**Reranking** (reorder results by relevance)
```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(top_n=3)
query_engine = index.as_query_engine(
    similarity_top_k=10,          # retrieve more initially
    node_postprocessors=[reranker] # then rerank to top 3
)
# Tradeoff: adds ~200ms latency + API cost, but significantly improves precision
```

**Metadata Filtering** (narrow search scope)
```python
# Add metadata during ingestion: section, page, date, category
# Filter at query time: "Only search the 'financials' section"
# Tradeoff: requires upfront metadata extraction, but dramatically improves relevance
```

---

## 7. Enterprise Patterns

### 7.1 Observability (Non-negotiable for Production)

```python
# observability/tracing.py
# Option 1: Arize Phoenix (open source, visual)
from llama_index.core import set_global_handler
set_global_handler("arize_phoenix")

# Option 2: OpenTelemetry (standard, integrates with existing infra)
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
```

**What to trace:**
- Every LLM call (prompt, response, tokens, latency)
- Every retrieval (query, results, scores)
- Agent reasoning steps (which tool chosen, why)
- End-to-end latency per query

**Tradeoff**: Phoenix is easiest to start (one line of code). OpenTelemetry is more standard if you already have Datadog/Grafana.

### 7.2 Evaluation Pipeline

```python
# evaluation/evaluator.py
# Golden test set: human-verified Q&A pairs
test_cases = [
    {
        "query": "What is the organization's mission?",
        "expected_answer": "To provide education access...",
        "source_page": 3,
    },
    # ... 20-50 test cases for MVP
]

# Metrics to track:
# - Faithfulness: Does the answer match the source?
# - Relevancy: Are retrieved chunks relevant to the query?
# - Correctness: Does the answer match expected answer?
# - Latency: Time to first token, total time
# - Cost: Total tokens used per query
```

**Build this early.** Every change you make (new chunking, new model, new prompt) — run your eval suite. Without this, you're flying blind.

### 7.3 Guardrails

```python
# guardrails/input_validator.py
class InputGuardrails:
    def validate(self, query: str) -> tuple[bool, str]:
        # 1. Check query length (prevent prompt injection via massive inputs)
        if len(query) > 2000:
            return False, "Query too long"

        # 2. Check for PII (don't let users accidentally paste sensitive data)
        if self.contains_pii(query):
            return False, "Query contains personal information"

        # 3. Check relevance (is this about our documents?)
        # Optional: use a classifier or simple keyword check
        return True, "OK"

# guardrails/output_validator.py
class OutputGuardrails:
    def validate(self, response: str, sources: list) -> tuple[bool, str]:
        # 1. Check for hallucination markers
        if not sources:
            return False, "No sources found — response may be hallucinated"

        # 2. Check response isn't just repeating the question
        # 3. Check for harmful/inappropriate content
        return True, "OK"
```

### 7.4 Error Handling & Resilience

```python
# Retry with exponential backoff for API calls
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def query_with_retry(agent, query):
    return agent.chat(query)

# Graceful degradation: if agent fails, fall back to simple vector search
def robust_query(agent, vector_engine, query):
    try:
        return agent.chat(query)
    except Exception as e:
        logger.warning(f"Agent failed: {e}, falling back to vector search")
        return vector_engine.query(query)
```

---

## 8. Implementation Roadmap

### Sprint 1 (Week 1-2): MVP Foundation
- [ ] Set up project structure and config management
- [ ] Implement PDF ingestion pipeline (PyMuPDF + SentenceSplitter)
- [ ] Set up ChromaDB vector store
- [ ] Build vector query engine + summary query engine
- [ ] Create ReActAgent with both tools
- [ ] Build interactive CLI for testing
- [ ] Write 10 golden test cases

### Sprint 2 (Week 3-4): MVP Hardening
- [ ] Add observability (Arize Phoenix)
- [ ] Build evaluation pipeline
- [ ] Add input/output guardrails
- [ ] Add error handling and retry logic
- [ ] Experiment with chunk sizes (run evals to compare)
- [ ] Add metadata extraction (page numbers, sections)
- [ ] Expand test cases to 30+

### Sprint 3 (Week 5-6): Multi-Document Foundation
- [ ] Refactor to support multiple documents
- [ ] Implement document agent pattern (1 agent per doc)
- [ ] Build top-level routing agent
- [ ] Add SubQuestionQueryEngine for cross-doc queries
- [ ] Evaluate: Qdrant vs ChromaDB for multi-doc

### Sprint 4 (Week 7-8): Production Readiness
- [ ] Add FastAPI REST API
- [ ] Implement hybrid search (BM25 + vector)
- [ ] Add reranking (Cohere or cross-encoder)
- [ ] Set up structured logging
- [ ] Load testing and optimization
- [ ] Documentation

### Sprint 5 (Week 9-10): Enterprise Polish
- [ ] Add authentication to API
- [ ] Implement rate limiting
- [ ] Set up CI/CD for eval pipeline
- [ ] Add conversation memory / chat history
- [ ] Multi-tenancy support (if needed)
- [ ] Deploy to cloud (Docker + orchestration)

---

## Key Decisions Summary

| Decision | MVP Choice | Why | Revisit When |
|----------|-----------|-----|-------------|
| PDF Parser | PyMuPDF | Simple, local, free | PDFs have complex tables/layouts |
| Chunking | SentenceSplitter (512) | Simple, proven | Eval shows retrieval quality issues |
| Embeddings | text-embedding-3-small | Cheap, good enough | Need better retrieval precision |
| Vector DB | ChromaDB | Zero setup | Need multi-tenancy or metadata filtering |
| Agent Type | ReActAgent | Transparent reasoning | Need lower latency (switch to OpenAIAgent) |
| LLM | GPT-4o | Best reasoning | Cost optimization → GPT-4o-mini for routing |
| Response Mode | compact | Balanced cost/quality | Need detailed answers → refine |
| Observability | Arize Phoenix | One-line setup | Need Grafana integration → OpenTelemetry |

---

## Next Steps

Ready to start building? The recommended order is:

1. **Set up the project** — Create the folder structure, install dependencies
2. **Build ingestion** — Get your PDF into ChromaDB
3. **Build query engines** — Test retrieval quality before adding the agent
4. **Add the agent** — Wire up tools and test reasoning
5. **Add observability** — See what's happening under the hood
6. **Evaluate** — Build golden test set, measure quality

Each step is independent and testable. Don't move to the next until the current one works well.
