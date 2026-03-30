"""
retriever.py — Hybrid retrieval: ChromaDB (semantic) + BM25 (keyword) + chunk expansion.

EnsembleRetriever fuses both signals via Reciprocal Rank Fusion (RRF).
Chunk expansion: after retrieval, sibling chunks from the same source+section
are automatically added so sibling content is never incomplete.
"""

import json
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = Path(__file__).parent / "chroma_db"
CHUNKS_FILE = Path(__file__).parent / "chunks.json"

EMBED_MODEL = "nomic-embed-text"
COLLECTION = "docrag"

# "BEAST" optimization: Keep it tight (TOP_K=4) to avoid context noise. 
# Total context (with siblings) will stay around 10-12 chunks.
TOP_K = 4


def expand_chunks(retrieved: list[Document], all_chunks: list[Document]) -> list[Document]:
    """
    For each retrieved chunk, add its immediate neighbors (±1 index) from
    the same source document. This ensures sibling chunks from the same
    section are never left out due to top-k cutoff.
    """
    # Build a lookup: (source, page_content) -> index in all_chunks
    content_to_idx: dict[str, int] = {
        f"{c.metadata.get('source', '')}||{c.page_content}": i
        for i, c in enumerate(all_chunks)
    }

    seen: set[str] = set()
    expanded: list[Document] = []

    def add(doc: Document):
        key = f"{doc.metadata.get('source', '')}||{doc.page_content}"
        if key not in seen:
            seen.add(key)
            expanded.append(doc)

    for doc in retrieved:
        add(doc)
        
        # SKIP expansion for synthetic/golden chunks — they are already complete.
        if doc.metadata.get("_synthetic"):
            continue

        key = f"{doc.metadata.get('source', '')}||{doc.page_content}"
        idx = content_to_idx.get(key)
        if idx is None:
            continue
        src = doc.metadata.get("source", "")
        # Add previous neighbor if same source
        if idx > 0 and all_chunks[idx - 1].metadata.get("source") == src:
            add(all_chunks[idx - 1])
        # Add next neighbor if same source
        if idx < len(all_chunks) - 1 and all_chunks[idx + 1].metadata.get("source") == src:
            add(all_chunks[idx + 1])

    return expanded


def load_retriever():
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("ChromaDB not found. Run 'uv run main.py ingest' first.")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.json not found. Run 'uv run main.py ingest' first.")

    raw = json.loads(CHUNKS_FILE.read_text())
    all_chunks = [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in raw]

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    chroma = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    semantic_retriever = chroma.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = TOP_K

    # Ensemble: prioritized BM25 at 0.4 for exact keyword hits on summary chunks.
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    class ExpandingRetriever:
        def invoke(self, query: str) -> list[Document]:
            docs = ensemble.invoke(query)
            return expand_chunks(docs, all_chunks)

        def __getattr__(self, name):
            return getattr(ensemble, name)

    return ExpandingRetriever()
