"""
chain.py — RAG chain wiring retriever → prompt → LLM.

Uses LangChain Expression Language (LCEL) for composability and streaming support.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from retriever import load_retriever

LLM_MODEL = "llama3.1:8b"

# num_ctx: expand context window so all retrieved chunks + answer fit (default is 2048 — too small)
# num_predict: max output tokens (-1 = unlimited, 1024 is safe for detailed answers)
LLM_OPTIONS = dict(
    temperature=0,
    num_ctx=32768,   # k=10 chunks × 2500 chars ≈ 6k tokens context — 32k gives plenty of headroom
    num_predict=-1,
)

# Grounded, citation-aware prompt — prevents hallucination outside the docs
RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the question using ONLY the context below. Do not add information that is not in the context.

- The answer may be spread across multiple chunks — collect all relevant items before answering.
- If listing items (types, steps, options), include every one you find. Do not stop early.
- Cite the source document and section for each item.
- If the answer is not in the context at all, say "Not found in the documents."

Context:
{context}

Question: {question}

Answer:"""
)


def format_docs_with_sources(docs) -> str:
    """Format retrieved chunks with source + section breadcrumb for the prompt."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        label = f"[Source: {source}" + (f" | Section: {section}" if section else "") + "]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain():
    """
    Returns (answer_chain, retriever) tuple.
    answer_chain takes {"context": str, "question": str} — no internal retrieval,
    so callers retrieve once and pass context directly (avoids double retrieval).
    """
    retriever = load_retriever()
    llm = ChatOllama(model=LLM_MODEL, **LLM_OPTIONS)

    # No retriever in chain — caller passes pre-retrieved context
    answer_chain = RAG_PROMPT | llm | StrOutputParser()

    return answer_chain, retriever
