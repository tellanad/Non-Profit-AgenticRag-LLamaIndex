"""
Ingestion script — Run this to parse, chunk, embed, and store a PDF.

Usage:
    uv run python scripts/ingest.py data/documents/your_file.pdf

WHY A SEPARATE SCRIPT?
- Ingestion is a one-time (or occasional) operation, separate from querying
- You can re-run it when documents change without touching the agent
- Makes CI/CD simpler: ingest in one step, deploy agent in another
"""

import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from config.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    setup_logging(settings.log_level)

    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/ingest.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        sys.exit(1)

    logger.info(f"Starting ingestion for: {pdf_path}")

    # TODO: Step 1 — Parse PDF (ingestion/pdf_parser.py)
    # TODO: Step 2 — Chunk into nodes (ingestion/chunker.py)
    # TODO: Step 3 — Embed and store (ingestion/embedder.py)

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()
