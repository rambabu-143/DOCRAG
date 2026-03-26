"""
retriever.py — Hybrid retrieval: semantic (ChromaDB MMR) + keyword (BM25).

Industrial-standard retrieval:
- MMR (Maximal Marginal Relevance): avoids returning redundant near-duplicate chunks
- BM25: classic TF-IDF sparse retrieval — great for exact keywords, acronyms, codes
- EnsembleRetriever: weighted fusion of both signals (60/40 semantic/keyword)
  This mirrors the approach used in production RAG systems (e.g. Elastic, Azure AI Search)
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

# How many chunks to return per retriever
TOP_K = 12
# MMR fetch_k: fetch more candidates, then diversify down to TOP_K
MMR_FETCH_K = TOP_K * 4
# MMR lambda: 1.0 = pure similarity, 0.0 = pure diversity. 0.7 = balanced.
MMR_LAMBDA = 0.7


def load_retriever() -> EnsembleRetriever:
    """
    Build a hybrid retriever combining:
      - Semantic (ChromaDB + MMR) for conceptual/paraphrase matching
      - Keyword (BM25) for exact term matching (codes, acronyms, names)
    """
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB not found. Run 'uv run main.py ingest' first."
        )
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "chunks.json not found. Run 'uv run main.py ingest' first."
        )

    # --- Semantic retriever (dense) ---
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    chroma = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name=COLLECTION,
    )
    semantic_retriever = chroma.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": TOP_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )

    # --- Keyword retriever (sparse) ---
    raw = json.loads(CHUNKS_FILE.read_text())
    docs = [
        Document(page_content=r["page_content"], metadata=r["metadata"])
        for r in raw
    ]
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = TOP_K

    # --- Hybrid ensemble ---
    # Weights: 60% semantic, 40% keyword
    # Tune these based on your domain: boost BM25 if your queries use exact codes/names
    return EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )
