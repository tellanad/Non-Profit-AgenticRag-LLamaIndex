"""
Interactive query script — Chat with the agent about your documents.

Usage:
    uv run python scripts/query.py

WHY A SEPARATE SCRIPT?
- Quick way to test the agent during development
- No API server overhead — direct interaction
- Will be replaced by FastAPI endpoint in Phase 2
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path so imports work — MUST come before other imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from config.logging_config import setup_logging, get_logger
from agent.rag_agent import create_agent

logger = get_logger(__name__)


async def main():
    setup_logging(settings.log_level)
    logger.info("Loading agent...")

    agent = create_agent(verbose=True)

    print("\n=== Agentic RAG — Non-Profit Document Assistant ===")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        # v0.14+ ReActAgent is Workflow-based and uses async .run()
        response = await agent.run(query)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
