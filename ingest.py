"""
ingest.py — Load PDFs, chunk, embed, and store in ChromaDB.

Industrial-standard approach:
- PyMuPDF for reliable text extraction (handles headers, columns, tables)
- RecursiveCharacterTextSplitter with semantic separators
- Stable chunk IDs via MD5 hash → safe to re-run (upsert, no duplicates)
- Metadata: source filename + page number preserved on every chunk
- Separate chunks.json for BM25 sparse retrieval
"""

import hashlib
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from rich.console import Console
from rich.progress import track

PDF_DIR = Path(__file__).parent
CHROMA_DIR = PDF_DIR / "chroma_db"
CHUNKS_FILE = PDF_DIR / "chunks.json"

EMBED_MODEL = "nomic-embed-text"
COLLECTION = "docrag"

# Industrial-standard chunking params
CHUNK_SIZE = 1000       # chars — balances context vs. precision
CHUNK_OVERLAP = 200     # 20% overlap ensures no context lost at boundaries

console = Console()


def load_pdfs(pdf_dir: Path) -> list[Document]:
    """Extract text page-by-page from all PDFs, preserving metadata."""
    docs: list[Document] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        console.print("[red]No PDF files found in directory.[/red]")
        sys.exit(1)

    for pdf_path in pdf_files:
        fitz_doc = fitz.open(str(pdf_path))
        total_pages = len(fitz_doc)

        for page_num in range(total_pages):
            page = fitz_doc[page_num]
            # "text" mode preserves reading order better than raw extraction
            text = page.get_text("text").strip()

            if not text:
                continue  # skip blank/image-only pages

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_num + 1,       # 1-indexed for humans
                    "total_pages": total_pages,
                },
            ))

        fitz_doc.close()
        console.print(f"  [green]✓[/green] {pdf_path.name} — {total_pages} pages")

    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents using RecursiveCharacterTextSplitter.

    Separator priority (tries each in order until chunk fits):
      1. Double newline  — paragraph boundary
      2. Single newline  — line boundary
      3. Period-space    — sentence boundary
      4. Single space    — word boundary
      5. Empty string    — character (last resort)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,   # stores char offset in metadata for traceability
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def make_chunk_id(chunk: Document) -> str:
    """
    Stable, deterministic ID per chunk.
    Combining content + source + page means re-ingesting is safe (upsert semantics).
    """
    key = (
        chunk.page_content
        + chunk.metadata.get("source", "")
        + str(chunk.metadata.get("page", ""))
        + str(chunk.metadata.get("start_index", ""))
    )
    return hashlib.md5(key.encode()).hexdigest()


def ingest():
    console.rule("[bold cyan]DocRAG Ingest[/bold cyan]")

    # 1. Load
    console.print("\n[bold]Loading PDFs...[/bold]")
    docs = load_pdfs(PDF_DIR)
    console.print(f"  Extracted [cyan]{len(docs)}[/cyan] pages total\n")

    # 2. Chunk
    console.print("[bold]Chunking...[/bold]")
    chunks = chunk_documents(docs)
    console.print(
        f"  [cyan]{len(chunks)}[/cyan] chunks "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n"
    )

    # 3. Save chunks for BM25 sparse retrieval
    console.print("[bold]Saving chunks for BM25 index...[/bold]")
    CHUNKS_FILE.write_text(json.dumps([
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in chunks
    ], indent=2))
    console.print(f"  Saved → {CHUNKS_FILE.name}\n")

    # 4. Embed + store
    console.print("[bold]Embedding and storing in ChromaDB...[/bold]")
    console.print(f"  Model: [cyan]{EMBED_MODEL}[/cyan] (via Ollama)")
    console.print(f"  This may take a few minutes for the first run...\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    ids = [make_chunk_id(c) for c in chunks]

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
        ids=ids,
    )

    console.rule("[bold green]Ingest Complete[/bold green]")
    console.print(
        f"  [green]✓[/green] {len(chunks)} chunks indexed into ChromaDB\n"
        f"  Run [cyan]uv run main.py query[/cyan] to start asking questions."
    )


if __name__ == "__main__":
    ingest()
