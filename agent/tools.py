"""
Agent Tools — Wraps query engines as tools the agent can pick from.

This is the bridge between Layer 2 (engines) and Layer 3 (agent).
Each tool has a name and description — the agent reads these descriptions
to decide which tool to use for a given question.

WHY DESCRIPTIONS MATTER:
- The agent is an LLM. It picks tools based on their descriptions.
- A vague description = agent picks the wrong tool = bad answer.
- A clear description = agent routes correctly = good answer.
- Think of descriptions as "instructions for the agent on when to use this."

TOOL PATTERN:
  QueryEngineTool wraps a query engine with:
  - name: short identifier (used in agent's reasoning logs)
  - description: when to use this tool (the agent reads this)
"""

from llama_index.core.tools import QueryEngineTool

from engines.vector_engine import get_vector_query_engine
from engines.summary_engine import get_summary_query_engine
from config.logging_config import get_logger

logger = get_logger(__name__)

def get_agent_tools() -> list[QueryEngineTool]:
    """Create and return all tools available to the agent.

    Returns:
        List of QueryEngineTools ready to be passed to the agent.
    """
    # ── Tool 1: Vector Search (specific questions) ─────────────────
    vector_engine = get_vector_query_engine()

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        name="vector_search",
        description=(
            "Useful for finding specific facts, policies, numbers, dates, "
            "names, or details within the document. Use this when the "
            "question asks about a particular topic, section, or data point. "
            "Example questions: 'What is the refund policy?', "
            "'How much was the budget?', 'Who is the director?'"
        ),
    )

    # ── Tool 2: Document Summary (broad questions) ─────────────────
    summary_engine = get_summary_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        name="document_summary",
        description=(
            "Useful for getting a high-level overview or summary of the "
            "entire document or large sections. Use this when the question "
            "is broad, asks for a summary, or asks 'what is this about'. "
            "Example questions: 'Summarize this document', "
            "'What are the main topics?', 'Give me an overview.'"
        ),
    )

    logger.info(f"Agent tools ready: {[t.metadata.name for t in [vector_tool, summary_tool]]}")

    return [vector_tool, summary_tool]