"""
retriever.py — Hybrid retrieval: ChromaDB MMR (semantic) + BM25 (keyword).

EnsembleRetriever fuses both signals via Reciprocal Rank Fusion (RRF):
- Semantic catches paraphrasing and conceptual matches
- BM25 catches exact keywords, acronyms, section names
- MMR ensures diversity (no near-duplicate chunks)
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

TOP_K = 8
MMR_FETCH_K = TOP_K * 4
MMR_LAMBDA = 0.7


def load_retriever() -> EnsembleRetriever:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError("ChromaDB not found. Run 'uv run main.py ingest' first.")
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError("chunks.json not found. Run 'uv run main.py ingest' first.")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    chroma = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    semantic_retriever = chroma.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA},
    )

    raw = json.loads(CHUNKS_FILE.read_text())
    docs = [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in raw]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = TOP_K

    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
