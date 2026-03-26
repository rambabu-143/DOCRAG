"""
ingest.py — Load PDFs with Docling, clean, chunk, embed, store in ChromaDB.

Industrial-standard pipeline:
- Docling: deep layout analysis → clean Markdown (removes headers/footers,
  structures tables, preserves headings hierarchy)
- MarkdownHeaderTextSplitter: chunks follow document structure (headings),
  not arbitrary character counts
- RecursiveCharacterTextSplitter: secondary split for oversized chunks
- Stable MD5 chunk IDs → safe to re-run (upsert, no duplicates)
- Metadata preserved: source filename + header breadcrumb on every chunk
"""

import hashlib
import json
import re
import sys
from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rich.console import Console

PDF_DIR = Path(__file__).parent
CHROMA_DIR = PDF_DIR / "chroma_db"
CHUNKS_FILE = PDF_DIR / "chunks.json"

EMBED_MODEL = "nomic-embed-text"
COLLECTION = "docrag"

# Secondary split size — catches any sections too large after header split
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Markdown heading levels to split on
HEADERS_TO_SPLIT = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

console = Console()


def clean_markdown(text: str) -> str:
    """
    Remove common PDF noise from Docling's markdown output:
    - Isolated page numbers (e.g. "- 12 -", "Page 12", "12\n")
    - Repetitive header/footer patterns
    - Excessive blank lines (3+ → 2)
    - Hyphenated line-break artifacts (re-join split words)
    """
    # Remove page number patterns
    text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)       # - 12 -
    text = re.sub(r'\nPage\s+\d+\s*(of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\d+\s*\n', '\n', text)                    # lone numbers on a line

    # Re-join hyphenated words broken across lines
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Normalize excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


def load_and_clean_pdfs(pdf_dir: Path):
    """
    Use DoclingLoader to convert each PDF to clean Markdown,
    then apply additional cleaning. Returns list of (filename, markdown) tuples.
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print("[red]No PDF files found.[/red]")
        sys.exit(1)

    results = []
    for pdf_path in pdf_files:
        console.print(f"  [dim]Processing:[/dim] {pdf_path.name}")
        loader = DoclingLoader(file_path=str(pdf_path))
        docs = loader.load()

        # Docling may return multiple docs per file — join them
        full_text = "\n\n".join(d.page_content for d in docs)
        cleaned = clean_markdown(full_text)

        results.append((pdf_path.name, cleaned))
        console.print(f"  [green]✓[/green] {pdf_path.name} — {len(cleaned):,} chars after cleaning")

    return results


def chunk_documents(filename: str, markdown: str):
    """
    Two-stage chunking:
    1. MarkdownHeaderTextSplitter — respects document structure
    2. RecursiveCharacterTextSplitter — handles oversized sections
    """
    from langchain_core.documents import Document

    # Stage 1: split by headings
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT,
        strip_headers=False,  # keep headers in chunk for context
    )
    header_chunks = header_splitter.split_text(markdown)

    # Stage 2: secondary size-based split for large sections
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_chunks = size_splitter.split_documents(header_chunks)

    # Inject source filename into every chunk's metadata
    for chunk in final_chunks:
        chunk.metadata["source"] = filename
        # Build a readable breadcrumb from header metadata
        breadcrumb = " > ".join(
            chunk.metadata[k]
            for k in ("h1", "h2", "h3", "h4")
            if chunk.metadata.get(k)
        )
        if breadcrumb:
            chunk.metadata["section"] = breadcrumb

    return final_chunks


def make_chunk_id(chunk) -> str:
    key = (
        chunk.page_content
        + chunk.metadata.get("source", "")
        + chunk.metadata.get("section", "")
    )
    return hashlib.md5(key.encode()).hexdigest()


def ingest():
    console.rule("[bold cyan]DocRAG Ingest (Docling)[/bold cyan]")

    # 1. Load + clean PDFs
    console.print("\n[bold]Loading & cleaning PDFs with Docling...[/bold]")
    console.print("  [dim](First run downloads layout models — may take a minute)[/dim]\n")
    pdf_data = load_and_clean_pdfs(PDF_DIR)

    # 2. Chunk
    console.print("\n[bold]Chunking by document structure...[/bold]")
    all_chunks = []
    for filename, markdown in pdf_data:
        chunks = chunk_documents(filename, markdown)
        all_chunks.extend(chunks)
        console.print(f"  [cyan]{len(chunks)}[/cyan] chunks ← {filename}")

    console.print(f"\n  Total: [cyan]{len(all_chunks)}[/cyan] chunks\n")

    # 3. Save for BM25
    console.print("[bold]Saving chunks for BM25 index...[/bold]")
    CHUNKS_FILE.write_text(json.dumps([
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in all_chunks
    ], indent=2))
    console.print(f"  Saved → {CHUNKS_FILE.name}\n")

    # 4. Embed + store
    console.print("[bold]Embedding and storing in ChromaDB...[/bold]")
    console.print(f"  Model: [cyan]{EMBED_MODEL}[/cyan] via Ollama\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    ids = [make_chunk_id(c) for c in all_chunks]

    # Wipe old collection and rebuild fresh
    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        console.print("  [dim]Cleared old ChromaDB collection[/dim]")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
        ids=ids,
    )

    console.rule("[bold green]Ingest Complete[/bold green]")
    console.print(
        f"  [green]✓[/green] {len(all_chunks)} clean chunks indexed\n"
        f"  Run [cyan]uv run main.py query[/cyan] to start."
    )


if __name__ == "__main__":
    ingest()
