import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging
from ingestion.pdf_parser import parse_pdf


setup_logging("DEBUG")

docs = parse_pdf("data/documents/sample.pdf")

print(f"\n--- Results ---")
print(f"Total documents: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n[Page {i+1}]")
    print(f"  Metadata: {doc.metadata}")
    print(f"  Text preview: {doc.text[:200]}...")