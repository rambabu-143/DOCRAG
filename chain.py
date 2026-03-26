"""
chain.py — RAG chain wiring retriever → prompt → LLM.

Uses LangChain Expression Language (LCEL) for composability and streaming support.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from retriever import load_retriever

LLM_MODEL = "llama3.2"

# num_ctx: expand context window so all retrieved chunks + answer fit (default is 2048 — too small)
# num_predict: max output tokens (-1 = unlimited, 1024 is safe for detailed answers)
LLM_OPTIONS = dict(
    temperature=0,
    num_ctx=8192,
    num_predict=1024,
)

# Grounded, citation-aware prompt — prevents hallucination outside the docs
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a precise technical assistant. Your job is to answer questions using ONLY the provided document context.

Rules:
- Answer strictly from the context. Do not guess or add outside knowledge.
- If the answer is not in the context, say: "I couldn't find that in the provided documents."
- Always cite the source document and page number for every claim.
- Be thorough and complete — do not cut your answer short. Cover all relevant details from the context.
- Use bullet points or numbered lists when the answer has multiple parts or steps.

---
Context:
{context}
---

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
    Returns (chain, retriever) tuple.
    Retriever is returned separately so callers can show sources.
    """
    retriever = load_retriever()
    llm = ChatOllama(model=LLM_MODEL, **LLM_OPTIONS)

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever
