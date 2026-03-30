"""
ingest.py — Load PDFs with pymupdf4llm (text + tables) and EasyOCR (images).

Fully local, fully free pipeline:
- pymupdf4llm   → converts PDF pages to clean markdown (text + tables)
- EasyOCR       → extracts text from images/screenshots (fast, no LLM needed)
- nomic-embed-text → embeds all chunks into ChromaDB

Image OCR results are cached in image_cache.json — re-runs skip unchanged images.
"""

import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # pymupdf
import easyocr
import numpy as np
import ollama
import pymupdf4llm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

PDF_DIR = Path(__file__).parent
CHROMA_DIR = PDF_DIR / "chroma_db"
CHUNKS_FILE = PDF_DIR / "chunks.json"
IMAGE_CACHE_FILE = PDF_DIR / "image_cache.json"
PDF_HASH_FILE = PDF_DIR / "pdf_hashes.json"

EMBED_MODEL = "nomic-embed-text"
COLLECTION = "docrag"

MAX_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
MAX_EMBED_CHARS = 5000  # hard safety cap before sending to nomic-embed-text

console = Console()


def extract_section_from_markdown(text: str) -> str:
    """Pull the last markdown heading from a chunk as the section label."""
    headers = re.findall(r'^#{1,4}\s+(.+)$', text, re.MULTILINE)
    if headers:
        heading = headers[-1].strip()
        return heading[:120] if len(heading) > 120 else heading
    return ""


def load_image_cache() -> dict[str, str]:
    """Load cached image descriptions from disk."""
    if IMAGE_CACHE_FILE.exists():
        try:
            return json.loads(IMAGE_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_image_cache(cache: dict[str, str]) -> None:
    IMAGE_CACHE_FILE.write_text(json.dumps(cache, indent=2))


def image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


_ocr_reader: easyocr.Reader | None = None

def get_ocr_reader() -> easyocr.Reader:
    """Lazy-load EasyOCR reader (downloads models on first use, cached after)."""
    global _ocr_reader
    if _ocr_reader is None:
        console.print("  [dim]Loading EasyOCR model (first run only)...[/dim]")
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def ocr_image(image_bytes: bytes, cache: dict[str, str], reader: easyocr.Reader) -> str:
    """Run EasyOCR on image bytes. Returns extracted text, uses cache to skip repeats."""
    key = image_hash(image_bytes)
    if key in cache:
        return cache[key]

    try:
        img = fitz.open(stream=image_bytes, filetype="image")
        pix = img[0].get_pixmap(dpi=150)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        results = reader.readtext(img_np, detail=0, paragraph=True)
        text = "\n".join(results).strip()

        cache[key] = text
        return text
    except Exception:
        return ""


def load_pdf_chunks(pdf_path: Path, cache: dict[str, str]) -> list[Document]:
    """
    Extract chunks from one PDF:
    - Text + tables via pymupdf4llm (per-page markdown)
    - Images via llava (if vision model is available)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    results: list[Document] = []

    # ── 1. Text + Tables (per-page markdown) ──────────────────────────────────
    pages_md = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    for page_data in pages_md:
        text = page_data.get("text", "").strip()
        page_num = page_data.get("metadata", {}).get("page", None)

        if not text or len(text) < 40:
            continue

        chunks = splitter.split_text(text) if len(text) > MAX_CHUNK_SIZE else [text]
        for chunk in chunks:
            if len(chunk.strip()) < 40:
                continue
            results.append(Document(
                page_content=chunk,
                metadata={
                    "source": pdf_path.name,
                    "section": extract_section_from_markdown(chunk),
                    "page": page_num,
                    "type": "text",
                }
            ))

    # ── 2. Images — OCR with EasyOCR in parallel ──────────────────────────────
    doc = fitz.open(str(pdf_path))
    seen_xrefs: set[int] = set()
    image_jobs: list[tuple[int, bytes]] = []

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        for img in page.get_images(full=True):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                base_image = doc.extract_image(xref)
                if base_image.get("width", 0) < 100 or base_image.get("height", 0) < 100:
                    continue
                image_jobs.append((page_num, base_image["image"]))
            except Exception:
                continue
    doc.close()

    if image_jobs:
        console.print(f"    [dim]OCR on {len(image_jobs)} images...[/dim]")
        # EasyOCR is not thread-safe — run sequentially with shared reader
        reader = get_ocr_reader()
        for page_num, image_bytes in image_jobs:
            try:
                text = ocr_image(image_bytes, cache, reader)
                if text and len(text) > 20:
                    results.append(Document(
                        page_content=f"[Image text on page {page_num}]\n{text}",
                        metadata={
                            "source": pdf_path.name,
                            "section": "",
                            "page": page_num,
                            "type": "image",
                        }
                    ))
            except Exception:
                continue

    return results




def pdf_file_hash(pdf_path: Path) -> str:
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()


def load_pdf_hashes() -> dict[str, str]:
    if PDF_HASH_FILE.exists():
        try:
            return json.loads(PDF_HASH_FILE.read_text())
        except Exception:
            pass
    return {}


def save_pdf_hashes(hashes: dict[str, str]) -> None:
    PDF_HASH_FILE.write_text(json.dumps(hashes, indent=2))


def make_chunk_id(chunk: Document, index: int) -> str:
    key = (
        str(index)
        + chunk.page_content
        + chunk.metadata.get("source", "")
        + chunk.metadata.get("section", "")
        + str(chunk.metadata.get("page", ""))
    )
    return hashlib.md5(key.encode()).hexdigest()


def ingest():
    console.rule("[bold cyan]DocRAG Ingest[/bold cyan]")

    console.print("[green]✓[/green] Image extraction via [cyan]EasyOCR[/cyan] (local, no API needed)")

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print("[red]No PDFs found.[/red]")
        sys.exit(1)

    # Load caches
    cache = load_image_cache()
    pdf_hashes = load_pdf_hashes()
    if len(cache):
        console.print(f"[dim]Image cache: {len(cache)} entries[/dim]")

    # Pre-load OCR reader once in main thread before any parallel work
    get_ocr_reader()

    console.print("\n[bold]Parsing PDFs...[/bold]")
    console.print("  [dim](text + tables via pymupdf4llm, images via EasyOCR)[/dim]\n")

    all_chunks: list[Document] = []
    changed_pdfs: list[str] = []
    unchanged_chunks: list[Document] = []

    # Separate changed vs unchanged PDFs
    saved_chunks: list[dict] = json.loads(CHUNKS_FILE.read_text()) if CHUNKS_FILE.exists() else []
    for pdf_path in pdf_files:
        current_hash = pdf_file_hash(pdf_path)
        if pdf_hashes.get(pdf_path.name) == current_hash and saved_chunks:
            console.print(f"  [dim]Unchanged:[/dim] {pdf_path.name} [green](skipped)[/green]")
            unchanged_chunks.extend([
                Document(page_content=c["page_content"], metadata=c["metadata"])
                for c in saved_chunks if c["metadata"].get("source") == pdf_path.name
            ])
        else:
            changed_pdfs.append(pdf_path.name)

    if not changed_pdfs:
        console.print("\n[green]All PDFs unchanged — nothing to re-index.[/green]")
        console.print("  Add a new PDF or modify an existing one to trigger re-ingest.\n")
        return

    # Parse changed PDFs in parallel
    def parse_pdf(pdf_path: Path) -> tuple[str, list[Document]]:
        chunks = load_pdf_chunks(pdf_path, cache)
        pdf_hashes[pdf_path.name] = pdf_file_hash(pdf_path)
        return pdf_path.name, chunks

    new_chunks: list[Document] = []
    with ThreadPoolExecutor(max_workers=min(len(changed_pdfs), 4)) as executor:
        futures = {executor.submit(parse_pdf, p): p for p in pdf_files if p.name in changed_pdfs}
        for future in as_completed(futures):
            name, chunks = future.result()
            text_count = sum(1 for c in chunks if c.metadata.get("type") == "text")
            img_count = sum(1 for c in chunks if c.metadata.get("type") == "image")
            console.print(
                f"  [green]✓[/green] {name} → "
                f"[cyan]{text_count}[/cyan] text chunks, "
                f"[cyan]{img_count}[/cyan] image descriptions"
            )
            new_chunks.extend(chunks)

    all_chunks = unchanged_chunks + new_chunks

    console.print(f"\n  Total: [cyan]{len(all_chunks)}[/cyan] chunks\n")

    # Persist caches
    save_pdf_hashes(pdf_hashes)
    save_image_cache(cache)

    # Save for BM25
    console.print("[bold]Saving chunks for BM25...[/bold]")
    CHUNKS_FILE.write_text(json.dumps([
        {"page_content": c.page_content, "metadata": c.metadata}
        for c in all_chunks
    ], indent=2))
    console.print(f"  Saved → {CHUNKS_FILE.name}\n")

    # Embed + store
    console.print("[bold]Embedding into ChromaDB...[/bold]")
    console.print(f"  Model: [cyan]{EMBED_MODEL}[/cyan] via Ollama\n")

    # Hard truncation — protect against oversized markdown tables/code blocks
    texts = [
        c.page_content[:MAX_EMBED_CHARS] if len(c.page_content) > MAX_EMBED_CHARS else c.page_content
        for c in all_chunks
    ]
    ids = [make_chunk_id(c, i) for i, c in enumerate(all_chunks)]

    # Batch embed with progress bar — sends chunks in batches of 32
    BATCH_SIZE = 32
    all_embeddings: list[list[float]] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Embedding[/bold cyan]"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} chunks"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(texts))
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i: i + BATCH_SIZE]
            result = ollama.embed(model=EMBED_MODEL, input=batch)
            all_embeddings.extend(result.embeddings)
            progress.advance(task, len(batch))

    if CHROMA_DIR.exists():
        import shutil
        shutil.rmtree(CHROMA_DIR)
        console.print("  [dim]Cleared old ChromaDB[/dim]")

    # Store pre-computed embeddings directly — no second embed call
    chroma = Chroma(
        collection_name=COLLECTION,
        embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
        persist_directory=str(CHROMA_DIR),
    )
    chroma._collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=texts,
        metadatas=[c.metadata for c in all_chunks],
    )

    console.rule("[bold green]Done[/bold green]")
    console.print(f"  [green]✓[/green] {len(all_chunks)} chunks indexed\n"
                  f"  Run [cyan]uv run main.py query[/cyan]")


if __name__ == "__main__":
    ingest()
