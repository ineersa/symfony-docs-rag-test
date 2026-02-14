#!/usr/bin/env python3
"""Interactive retrieval REPL for Symfony docs RAG.

Queries ChromaDB with CodeRankEmbed-style query prefixes and
displays top-K results with citations.
"""

import argparse
import sys
from pathlib import Path

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_COLLECTION = "symfony_docs"
DEFAULT_TOP_K = 5
CHROMA_DIR = Path("data/chroma")

# CodeRankEmbed bi-encoder: queries need this prefix
QUERY_PREFIX = "Represent this query for searching relevant code: "


def embed_query(client: OpenAI, query: str, model: str = "CodeRankEmbed") -> list[float]:
    """Embed a query with the required task prefix."""
    # Add Symfony context if not mentioned
    enriched = query
    if "symfony" not in query.lower():
        enriched = f"{query} (Symfony PHP framework)"

    prefixed = f"{QUERY_PREFIX}{enriched}"
    response = client.embeddings.create(input=[prefixed], model=model)
    return response.data[0].embedding


def display_results(console: Console, query: str, results: dict, top_k: int):
    """Pretty-print retrieval results."""
    console.print()
    console.print(Panel(
        f"[bold]{query}[/]",
        title="Query",
        border_style="blue",
    ))

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        console.print("[yellow]No results found.[/]")
        return

    for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances), 1):
        source = meta.get("source", "?")
        breadcrumb = meta.get("breadcrumb", "")
        part = meta.get("part", "")

        # Build header
        header = f"[bold cyan]#{rank}[/]  "
        header += f"[dim]score:[/] [{'green' if dist < 0.3 else 'yellow' if dist < 0.5 else 'red'}]{1 - dist:.4f}[/]"
        header += f"  [dim]dist:[/] {dist:.4f}"

        # Source info
        source_info = f"[bold]{source}[/]"
        if breadcrumb:
            source_info += f"  →  [dim]{breadcrumb}[/]"
        if part:
            source_info += f"  [dim](part {part})[/]"

        # Truncate long documents for display
        display_doc = doc
        if len(display_doc) > 800:
            display_doc = display_doc[:800] + "\n[dim]... (truncated)[/]"

        console.print(Panel(
            f"{source_info}\n\n{display_doc}",
            title=header,
            border_style="dim",
            padding=(0, 1),
        ))

    console.print(f"[dim]Showing {len(ids)}/{top_k} results[/]")


def main():
    parser = argparse.ArgumentParser(description="Retrieve Symfony docs by similarity")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Embeddings API base URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR, help="ChromaDB storage directory")
    parser.add_argument("query", nargs="*", help="One-shot query (skip REPL)")
    args = parser.parse_args()

    console = Console()

    # Load ChromaDB
    if not args.chroma_dir.exists():
        console.print(f"[red]Error:[/] {args.chroma_dir} not found. Run embed.py first.")
        sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=str(args.chroma_dir))
    try:
        collection = chroma_client.get_collection(args.collection)
    except Exception:
        console.print(f"[red]Error:[/] Collection '{args.collection}' not found. Run embed.py first.")
        sys.exit(1)

    doc_count = collection.count()
    console.print(f"[bold]Loaded collection[/] '{args.collection}' with [cyan]{doc_count}[/] documents")

    # Setup embeddings client
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # One-shot mode
    if args.query:
        query = " ".join(args.query)
        query_embedding = embed_query(client, query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=args.top_k,
            include=["documents", "metadatas", "distances"],
        )
        display_results(console, query, results, args.top_k)
        return

    # Interactive REPL
    console.print(f"\n[bold]Symfony Docs Retrieval REPL[/]")
    console.print(f"Type a question to search. Commands: [dim]:quit, :top N, :help[/]\n")

    top_k = args.top_k

    while True:
        try:
            query = console.input("[bold green]query>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/]")
            break

        if not query:
            continue

        if query == ":quit" or query == ":q":
            console.print("[dim]Bye![/]")
            break

        if query == ":help" or query == ":h":
            console.print("[dim]Commands:[/]")
            console.print("  :quit / :q     — exit")
            console.print("  :top N         — set number of results (current: {top_k})")
            console.print("  :help / :h     — this help")
            continue

        if query.startswith(":top "):
            try:
                top_k = int(query.split()[1])
                console.print(f"[dim]Top-K set to {top_k}[/]")
            except (IndexError, ValueError):
                console.print("[red]Usage: :top N[/]")
            continue

        # Query
        try:
            query_embedding = embed_query(client, query)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            display_results(console, query, results, top_k)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")


if __name__ == "__main__":
    main()
