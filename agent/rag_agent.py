"""
RAG Agent — The brain that decides how to answer questions.

This is Layer 3 — the orchestration layer. The agent receives a user
question and DECIDES what to do:
- Use vector_search for specific fact-finding
- Use document_summary for broad overview questions
- Combine multiple tool calls for complex questions
- Say "I don't know" if no relevant info is found

HOW ReActAgent WORKS:
  ReAct = Reason + Act. The agent follows a loop:
  1. THOUGHT: "The user is asking about budget details, I should search"
  2. ACTION: Calls vector_search tool with the query
  3. OBSERVATION: Gets back the retrieved chunks + synthesized answer
  4. THOUGHT: "This answers the question sufficiently"
  5. ANSWER: Returns the final response to the user

NOTE: LlamaIndex v0.14+ moved ReActAgent to a Workflow-based architecture.
  - Import from llama_index.core.agent.workflow (not llama_index.core.agent)
  - Use direct constructor ReActAgent() (not from_tools())
  - system_prompt is now called extra_context
  - The agent uses .run() for single queries (async)
"""

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI

from agent.tools import get_agent_tools
from config import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


def create_agent(verbose: bool = True) -> ReActAgent:
    """Create and return the RAG agent with all tools.

    Args:
        verbose: If True, enables detailed logging of agent reasoning.

    Returns:
        ReActAgent ready to answer questions about documents.
    """
    # ── Create the LLM ─────────────────────────────────────────────
    llm = OpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    # ── Load all tools ─────────────────────────────────────────────
    tools = get_agent_tools()

    logger.info(
        f"Creating agent with {len(tools)} tools: "
        f"{[t.metadata.name for t in tools]}"
    )

    # ── Create the ReAct Agent ─────────────────────────────────────
    agent = ReActAgent(
        tools=tools,
        llm=llm,
        extra_context=settings.agent_system_prompt,
    )

    logger.info(
        f"Agent ready: model={settings.llm_model}, "
        f"max_iterations={settings.max_agent_iterations}"
    )

    return agent
