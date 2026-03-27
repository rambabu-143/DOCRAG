"""
main.py — CLI entry point for DocRAG.

Commands:
  uv run main.py ingest        — Index all PDFs into ChromaDB
  uv run main.py query         — Interactive Q&A loop
  uv run main.py query "..."   — Ask a single question and exit
"""

import argparse
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

console = Console()

BANNER = """[bold cyan]
  ██████╗  ██████╗  ██████╗ ██████╗  █████╗  ██████╗
  ██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗██╔══██╗██╔════╝
  ██║  ██║██║   ██║██║      ██████╔╝███████║██║  ███╗
  ██║  ██║██║   ██║██║      ██╔══██╗██╔══██║██║   ██║
  ██████╔╝╚██████╔╝╚██████╗ ██║  ██║██║  ██║╚██████╔╝
  ╚═════╝  ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
[/bold cyan]"""


def cmd_ingest():
    from ingest import ingest
    ingest()


def cmd_query(question: str | None = None):
    console.print(BANNER)
    console.print(Panel(
        "[bold]PDF Question Answering[/bold]\n"
        "[dim]Hybrid RAG · llama3.2 · ChromaDB · BM25[/dim]",
        expand=False,
        border_style="cyan",
    ))

    console.print("[dim]Loading retriever and chain...[/dim]")
    try:
        from chain import build_chain
        chain, retriever = build_chain()
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print("[green]Ready![/green] Type your question (or 'quit' to exit)\n")

    if question:
        _ask(chain, retriever, question)
        return

    while True:
        try:
            question = Prompt.ask("[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower().strip() in ("quit", "exit", "q", "bye"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if not question.strip():
            continue

        _ask(chain, retriever, question)


def _ask(chain, retriever, question: str):
    # 1. Retrieve once
    with console.status("[bold yellow]Searching docs...[/bold yellow]"):
        try:
            docs = retriever.invoke(question)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            return

    from chain import format_docs_with_sources
    context = format_docs_with_sources(docs)

    # 2. Stream the answer — bypass Rich buffer so every chunk prints immediately
    console.print()
    console.rule("[bold blue]Answer[/bold blue]", style="blue")
    answer_parts: list[str] = []
    try:
        for chunk in chain.stream({"context": context, "question": question}):
            print(chunk, end="", flush=True)  # flush=True ensures no buffering cutoff
            answer_parts.append(chunk)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        return
    import sys; sys.stdout.flush()
    console.print()
    console.rule(style="blue")

    # Print sources table
    seen: set[tuple] = set()

    if docs:
        table = Table(title="Sources", show_header=True, header_style="bold dim")
        table.add_column("Document", style="cyan")
        table.add_column("Section", style="dim")
        for doc in docs:
            src = doc.metadata.get("source", "?")
            section = doc.metadata.get("section", "—")
            key = (src, section)
            if key not in seen:
                seen.add(key)
                table.add_row(src, section)
        console.print(table)

    console.print()


def main():
    parser = argparse.ArgumentParser(
        prog="docrag",
        description="Industrial-grade RAG over PDF documents",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Parse PDFs and index into ChromaDB")

    query_parser = sub.add_parser("query", help="Ask questions about the PDFs")
    query_parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Optional: ask a single question and exit",
    )

    args = parser.parse_args()

    if args.cmd == "ingest":
        cmd_ingest()
    elif args.cmd == "query":
        cmd_query(getattr(args, "question", None))


if __name__ == "__main__":
    main()
