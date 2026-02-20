"""
Interactive query script — Chat with the agent about your documents.

Usage:
    uv run python scripts/query.py

WHY A SEPARATE SCRIPT?
- Quick way to test the agent during development
- No API server overhead — direct interaction
- Will be replaced by FastAPI endpoint in Phase 2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from config.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    setup_logging(settings.log_level)
    logger.info("Loading agent...")

    # TODO: Load the agent from agent/rag_agent.py

    print("\n=== Agentic RAG — Non-Profit Document Assistant ===")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        # TODO: response = agent.chat(query)
        # TODO: print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
