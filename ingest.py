"""
ingest.py — Load PDFs with Docling, filter noise, embed, store in ChromaDB.

What was wrong before:
- Docling already returns semantic chunks (206 for a 61-page doc)
- We were joining them ALL back into one blob and re-splitting — destroying structure
- TOC entries (full of "......." dots) were indexed as real content

Fixed pipeline:
- Use Docling's own HierarchicalChunker chunks directly
- Filter noise: TOC dots, blank chunks, page-number-only lines
- Extract section heading from Docling's dl_meta for rich metadata
- RecursiveCharacterTextSplitter only as safety net for oversized chunks
- Stable MD5 IDs → safe to re-run
"""

import hashlib
import json
import re
import sys
from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rich.console import Console

PDF_DIR = Path(__file__).parent
CHROMA_DIR = PDF_DIR / "chroma_db"
CHUNKS_FILE = PDF_DIR / "chunks.json"

EMBED_MODEL = "nomic-embed-text"
COLLECTION = "docrag"

# Only split further if Docling chunk exceeds this
MAX_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

console = Console()


# ── Noise filters ─────────────────────────────────────────────────────────────

def is_noise(text: str) -> bool:
    """Return True if this chunk is TOC/header/footer noise, not real content."""
    stripped = text.strip()

    if len(stripped) < 30:
        return True  # too short to be useful

    # TOC lines: lots of dots (table of contents)
    dot_ratio = stripped.count('.') / max(len(stripped), 1)
    if dot_ratio > 0.3:
        return True

    # Pure page number lines
    if re.fullmatch(r'[\d\s\-–|/]+', stripped):
        return True

    # Only whitespace / dashes
    if re.fullmatch(r'[\s\-_=*]+', stripped):
        return True

    return False


def clean_text(text: str) -> str:
    """Light cleaning: fix hyphenation, normalize whitespace."""
    # Re-join hyphenated line breaks
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    # Collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── Metadata helpers ───────────────────────────────────────────────────────────

def extract_section(doc: Document) -> str:
    """Pull heading breadcrumb from Docling's dl_meta."""
    try:
        headings = doc.metadata.get("dl_meta", {}).get("headings", [])
        if headings:
            # Take the last (most specific) heading, trim if too long
            heading = headings[-1].strip()
            return heading[:120] if len(heading) > 120 else heading
    except Exception:
        pass
    return ""


def extract_page(doc: Document) -> int | None:
    """Extract page number from Docling's provenance metadata."""
    try:
        items = doc.metadata.get("dl_meta", {}).get("doc_items", [])
        if items:
            prov = items[0].get("prov", [])
            if prov:
                return prov[0].get("page_no")
    except Exception:
        pass
    return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

def load_pdf_chunks(pdf_path: Path) -> list[Document]:
    """
    Load one PDF via Docling. Returns its semantic chunks,
    filtered and cleaned, with rich metadata.
    """
    loader = DoclingLoader(file_path=str(pdf_path))
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    results = []
    for doc in raw_docs:
        text = clean_text(doc.page_content)

        if is_noise(text):
            continue

        # If Docling chunk is still too large, split further
        if len(text) > MAX_CHUNK_SIZE:
            sub_chunks = splitter.split_documents([Document(
                page_content=text,
                metadata=doc.metadata,
            )])
            docs_to_add = sub_chunks
        else:
            docs_to_add = [Document(page_content=text, metadata=doc.metadata)]

        for d in docs_to_add:
            section = extract_section(d)
            page = extract_page(d)
            d.metadata = {
                "source": pdf_path.name,
                "section": section,
                "page": page,
            }
            results.append(d)

    return results


def make_chunk_id(chunk: Document) -> str:
    key = (
        chunk.page_content
        + chunk.metadata.get("source", "")
        + chunk.metadata.get("section", "")
        + str(chunk.metadata.get("page", ""))
    )
    return hashlib.md5(key.encode()).hexdigest()


def ingest():
    console.rule("[bold cyan]DocRAG Ingest[/bold cyan]")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print("[red]No PDFs found.[/red]")
        sys.exit(1)

    # 1. Load + filter + clean
    console.print("\n[bold]Loading PDFs with Docling...[/bold]")
    console.print("  [dim](First run may download layout models)[/dim]\n")

    all_chunks: list[Document] = []
    for pdf_path in pdf_files:
        console.print(f"  [dim]Processing:[/dim] {pdf_path.name}")
        chunks = load_pdf_chunks(pdf_path)
        all_chunks.extend(chunks)
        console.print(f"  [green]✓[/green] {pdf_path.name} → [cyan]{len(chunks)}[/cyan] clean chunks")

    console.print(f"\n  Total: [cyan]{len(all_chunks)}[/cyan] chunks\n")

    # 2. Save for BM25
    console.print("[bold]Saving chunks for BM25...[/bold]")
    CHUNKS_FILE.write_text(json.dumps([
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in all_chunks
    ], indent=2))
    console.print(f"  Saved → {CHUNKS_FILE.name}\n")

    # 3. Embed + store
    console.print("[bold]Embedding into ChromaDB...[/bold]")
    console.print(f"  Model: [cyan]{EMBED_MODEL}[/cyan] via Ollama\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    ids = [make_chunk_id(c) for c in all_chunks]

    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        console.print("  [dim]Cleared old ChromaDB[/dim]")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
        ids=ids,
    )

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  [green]✓[/green] {len(all_chunks)} chunks indexed\n"
                  f"  Run [cyan]uv run main.py query[/cyan]")


if __name__ == "__main__":
    ingest()
