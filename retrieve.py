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

from bm25_common import BM25Index, reciprocal_rank_fusion

# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_COLLECTION = "symfony_docs"
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_CANDIDATES = 40
DEFAULT_BM25_CANDIDATES = 40
DEFAULT_RRF_K = 60
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
    parser.add_argument("--vector-candidates", type=int, default=DEFAULT_VECTOR_CANDIDATES, help="Vector shortlist size before RRF")
    parser.add_argument("--bm25-candidates", type=int, default=DEFAULT_BM25_CANDIDATES, help="BM25 shortlist size before RRF")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant")
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

    corpus = collection.get(include=["documents", "metadatas"])
    corpus_ids = [str(x) for x in (corpus.get("ids") or [])]
    corpus_docs = list(corpus.get("documents") or [])
    corpus_metas = list(corpus.get("metadatas") or [])
    id_to_row = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}

    bm25_texts: list[str] = []
    for doc, meta in zip(corpus_docs, corpus_metas):
        m = dict(meta or {})
        bm25_texts.append(
            "\n".join(
                [
                    str(m.get("source", "")),
                    str(m.get("breadcrumb", "")),
                    str(doc or ""),
                ]
            ).strip()
        )
    bm25_index = BM25Index(bm25_texts)

    # Setup embeddings client
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # One-shot mode
    if args.query:
        query = " ".join(args.query)
        query_embedding = embed_query(client, query)
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(args.top_k, args.vector_candidates),
            include=["documents", "metadatas", "distances"],
        )
        ids_raw = vector_results.get("ids") or [[]]
        docs_raw = vector_results.get("documents") or [[]]
        metas_raw = vector_results.get("metadatas") or [[]]
        dists_raw = vector_results.get("distances") or [[]]
        vector_ids = [str(x) for x in (ids_raw[0] if ids_raw else [])]
        vector_docs = docs_raw[0] if docs_raw else []
        vector_metas = metas_raw[0] if metas_raw else []
        vector_distances = dists_raw[0] if dists_raw else []

        vector_by_id: dict[str, tuple[str, dict, float]] = {}
        for doc_id, doc, meta, dist in zip(vector_ids, vector_docs, vector_metas, vector_distances):
            vector_by_id[doc_id] = (str(doc or ""), dict(meta or {}), float(dist) if dist is not None else 1.0)

        bm25_hits = bm25_index.search(query, args.bm25_candidates)
        bm25_ids = [corpus_ids[h.index] for h in bm25_hits if 0 <= h.index < len(corpus_ids)]

        fused = reciprocal_rank_fusion([vector_ids, bm25_ids], rrf_k=args.rrf_k)
        top_ids = [doc_id for doc_id, _ in fused[: args.top_k]]

        out_docs: list[str] = []
        out_metas: list[dict] = []
        out_distances: list[float] = []
        for doc_id in top_ids:
            if doc_id in vector_by_id:
                doc, meta, dist = vector_by_id[doc_id]
                out_docs.append(doc)
                out_metas.append(meta)
                out_distances.append(dist)
                continue
            idx = id_to_row[doc_id]
            meta = dict(corpus_metas[idx] or {})
            out_docs.append(str(corpus_docs[idx] or ""))
            out_metas.append(meta)
            out_distances.append(0.99)

        results = {
            "ids": [top_ids],
            "documents": [out_docs],
            "metadatas": [out_metas],
            "distances": [out_distances],
        }
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
            vector_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max(top_k, args.vector_candidates),
                include=["documents", "metadatas", "distances"],
            )
            ids_raw = vector_results.get("ids") or [[]]
            docs_raw = vector_results.get("documents") or [[]]
            metas_raw = vector_results.get("metadatas") or [[]]
            dists_raw = vector_results.get("distances") or [[]]
            vector_ids = [str(x) for x in (ids_raw[0] if ids_raw else [])]
            vector_docs = docs_raw[0] if docs_raw else []
            vector_metas = metas_raw[0] if metas_raw else []
            vector_distances = dists_raw[0] if dists_raw else []

            vector_by_id: dict[str, tuple[str, dict, float]] = {}
            for doc_id, doc, meta, dist in zip(vector_ids, vector_docs, vector_metas, vector_distances):
                vector_by_id[doc_id] = (str(doc or ""), dict(meta or {}), float(dist) if dist is not None else 1.0)

            bm25_hits = bm25_index.search(query, args.bm25_candidates)
            bm25_ids = [corpus_ids[h.index] for h in bm25_hits if 0 <= h.index < len(corpus_ids)]

            fused = reciprocal_rank_fusion([vector_ids, bm25_ids], rrf_k=args.rrf_k)
            top_ids = [doc_id for doc_id, _ in fused[:top_k]]

            out_docs: list[str] = []
            out_metas: list[dict] = []
            out_distances: list[float] = []
            for doc_id in top_ids:
                if doc_id in vector_by_id:
                    doc, meta, dist = vector_by_id[doc_id]
                    out_docs.append(doc)
                    out_metas.append(meta)
                    out_distances.append(dist)
                    continue
                idx = id_to_row[doc_id]
                out_docs.append(str(corpus_docs[idx] or ""))
                out_metas.append(dict(corpus_metas[idx] or {}))
                out_distances.append(0.99)

            results = {
                "ids": [top_ids],
                "documents": [out_docs],
                "metadatas": [out_metas],
                "distances": [out_distances],
            }
            display_results(console, query, results, top_k)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")


if __name__ == "__main__":
    main()
