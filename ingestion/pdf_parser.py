"""
PDF Parser — Loads PDF files into LlamaIndex Document objects with enriched metadata.

Uses SimpleDirectoryReader for text extraction, then enriches each Document
with section/heading structure extracted via PyMuPDF (fitz).

METADATA ENRICHMENT STRATEGY:
1. Extract TOC (Table of Contents) from PDF bookmarks if available
2. Map each page to its parent section(s) using TOC page ranges
3. If no TOC exists, fall back to font-size-based heading detection
4. Inject section info into each Document's metadata

WHY THIS MATTERS FOR RAG:
- Chunks know which section they belong to ("Financial Policies" vs "Mission Statement")
- Agent can cite sections, not just page numbers
- Enables metadata filtering: "only search the Budget section"
- Dramatically improves retrieval relevance for structured documents
"""

from pathlib import Path

import fitz  # PyMuPDF

from llama_index.core import SimpleDirectoryReader, Document

from config.logging_config import get_logger

logger = get_logger(__name__)


def _extract_toc_sections(pdf_path: str) -> dict[int, dict]:
    """Extract section structure from PDF Table of Contents (bookmarks).

    The TOC gives us [level, title, page_number] entries. We convert this
    into a mapping of page_number -> section info by determining which
    section each page falls under.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping page number (1-indexed) to section metadata:
        {
            1: {"section": "Introduction", "section_level": 1},
            5: {"section": "Financial Policies", "section_level": 1},
            ...
        }
        Returns empty dict if no TOC found.
    """
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()  # list of [level, title, page_number]
    total_pages = doc.page_count
    doc.close()

    if not toc:
        return {}

    logger.info(f"Found TOC with {len(toc)} entries")

    # Build page-to-section mapping
    # Each TOC entry applies from its page until the next entry's page
    page_sections: dict[int, dict] = {}

    for i, (level, title, page_num) in enumerate(toc):
        # Determine the page range this section covers
        if i + 1 < len(toc):
            next_page = toc[i + 1][2]
        else:
            next_page = total_pages + 1

        # Assign this section to all pages in its range
        for page in range(page_num, next_page):
            # Only assign if no section set yet, or if this is a deeper level
            # (deeper levels are more specific, so they take priority)
            if page not in page_sections or level > page_sections[page]["section_level"]:
                page_sections[page] = {
                    "section": title.strip(),
                    "section_level": level,
                }

    return page_sections


def _detect_headings_by_font(pdf_path: str) -> dict[int, dict]:
    """Fallback: detect section headings by analyzing font sizes.

    Scans each page for text blocks with larger-than-average font sizes,
    treating them as section headings. Less reliable than TOC but works
    for PDFs without bookmarks.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict mapping page number (1-indexed) to detected section metadata.
        Returns empty dict if no headings detected.
    """
    doc = fitz.open(pdf_path)
    page_sections: dict[int, dict] = {}
    current_heading = "Untitled Section"

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        # Collect all font sizes on this page to find the average
        font_sizes = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        font_sizes.append(span["size"])

        if not font_sizes:
            continue

        avg_font_size = sum(font_sizes) / len(font_sizes)

        # Look for text significantly larger than average (likely a heading)
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    # Heading heuristic: font size > 1.2x average AND short text
                    if (
                        text
                        and span["size"] > avg_font_size * 1.2
                        and len(text) < 100  # headings are short
                    ):
                        current_heading = text
                        break

        # 1-indexed page number to match SimpleDirectoryReader's page_label
        page_sections[page_num + 1] = {
            "section": current_heading,
            "section_level": 1,  # can't determine depth from font size alone
        }

    doc.close()
    return page_sections


def _extract_doc_metadata(pdf_path: str) -> dict:
    """Extract document-level metadata from PDF properties.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with document-level metadata (title, author, total_pages, etc.)
    """
    doc = fitz.open(pdf_path)
    metadata = doc.metadata or {}
    total_pages = doc.page_count
    doc.close()

    return {
        "total_pages": total_pages,
        "pdf_title": metadata.get("title", "").strip() or None,
        "pdf_author": metadata.get("author", "").strip() or None,
        "pdf_creation_date": metadata.get("creationDate", "").strip() or None,
    }


def parse_pdf(file_path: str | Path) -> list[Document]:
    """Parse a PDF file and return Documents with enriched metadata.

    Each Document contains the text from one page plus metadata:
    - file_name: original filename (from SimpleDirectoryReader)
    - page_label: page number string (from SimpleDirectoryReader)
    - section: heading/section this page belongs to (enriched)
    - section_level: depth of the heading, 1 = top level (enriched)
    - total_pages: total pages in the PDF (enriched)
    - pdf_title: document title from PDF properties (enriched)
    - pdf_author: document author from PDF properties (enriched)

    Args:
        file_path: Path to the PDF file to parse.

    Returns:
        List of Document objects, one per page, with enriched metadata.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a PDF.
    """
    file_path = Path(file_path)

    # ── Validation ──────────────────────────────────────────────────────
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    # ── Parse text via SimpleDirectoryReader ────────────────────────────
    logger.info(f"Parsing PDF: {file_path.name}")

    reader = SimpleDirectoryReader(input_files=[str(file_path)])
    documents = reader.load_data()

    # Filter out empty documents (blank pages)
    documents = [doc for doc in documents if doc.text.strip()]

    # ── Extract section structure ───────────────────────────────────────
    pdf_path_str = str(file_path)

    # Try TOC first (reliable), fall back to font-based detection
    page_sections = _extract_toc_sections(pdf_path_str)

    if page_sections:
        logger.info("Using TOC-based section detection")
    else:
        logger.info("No TOC found — falling back to font-based heading detection")
        page_sections = _detect_headings_by_font(pdf_path_str)

    # ── Extract document-level metadata ─────────────────────────────────
    doc_metadata = _extract_doc_metadata(pdf_path_str)

    # ── Enrich each Document's metadata ─────────────────────────────────
    for doc in documents:
        # Get page number from SimpleDirectoryReader's metadata
        page_label = doc.metadata.get("page_label", "1")
        try:
            page_num = int(page_label)
        except ValueError:
            page_num = 1

        # Add section info
        section_info = page_sections.get(page_num, {})
        doc.metadata["section"] = section_info.get("section", "Unknown Section")
        doc.metadata["section_level"] = section_info.get("section_level", 0)

        # Add document-level metadata
        doc.metadata.update(doc_metadata)

    logger.info(
        f"Parsed {file_path.name}: "
        f"{len(documents)} pages with content, "
        f"sections detected: {len(set(s.get('section', '') for s in page_sections.values()))}"
    )

    return documents
