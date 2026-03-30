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

# num_ctx: 8192 — Maximum overhead but zero truncation risk.
# Confirmed safe for 16GB systems with 6.5GB free (wmic measurement).
LLM_OPTIONS = dict(
    temperature=0,
    num_ctx=8192,
    num_predict=1024,
)

# BEAST MODE PROMPT: Strict, aggressive, and exhaustive.
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior document auditor. Answer using ONLY the provided context.

YOUR PRIMARY SOURCE IS THE "SDX Virtual Folder Types Summary" CHUNK.

EXHAUSTIVE ENUMERATION RULES:
1. If asked for types or a list, you MUST enumerate EVERY item you find in the context.
2. If the summary chunk lists 13 items, you MUST output 13 items.
3. Use a numbered list (1, 2, 3...) for clarity.
4. Do NOT stop early. Failure to list every item is unacceptable.
5. Do NOT add preamble — start the list immediately if possible.

Context:
{context}

Question: {question}

Answer (ENUMERATE EVERY SINGLE ITEM FOUND):"""
)


def format_docs_with_sources(docs) -> str:
    """
    Format retrieved chunks with source + section breadcrumb.
    Synthetic/Summary chunks are sorted to the TOP so the LLM sees them first.
    """
    # Sort: Synthetic chunks first, then by score/rank (original order)
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get("_synthetic", False), reverse=True)
    
    parts = []
    for doc in sorted_docs:
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        label = f"[Source: {source}" + (f" | Section: {section}" if section else "") + "]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_chain():
    """
    Returns (answer_chain, retriever) tuple.
    """
    retriever = load_retriever()
    llm = ChatOllama(model=LLM_MODEL, **LLM_OPTIONS)

    # No retriever in chain — caller passes pre-retrieved context
    answer_chain = RAG_PROMPT | llm | StrOutputParser()

    return answer_chain, retriever
