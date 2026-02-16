#!/usr/bin/env python3
"""Interactive retrieval REPL for Symfony docs RAG.

Queries ChromaDB with CodeRankEmbed-style query prefixes and
displays top-K results with citations.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from bm25_common import BM25Index, reciprocal_rank_fusion
from hyde_common import (
    DEFAULT_HYDE_BASE_URL,
    DEFAULT_HYDE_MAX_CHARS,
    DEFAULT_HYDE_MODEL,
    DEFAULT_HYDE_TEMPERATURE,
    DEFAULT_HYDE_VARIANT_WEIGHT,
    DEFAULT_HYDE_VARIANTS,
    HyDEGenerator,
)
from rerank_common import (
    BGEReranker,
    DEFAULT_RERANKER_DEVICE,
    DEFAULT_RERANKER_MODEL,
    build_rerank_passage,
)

# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_COLLECTION = "symfony_docs"
DEFAULT_TOP_K = 5
DEFAULT_VECTOR_CANDIDATES = 40
DEFAULT_BM25_CANDIDATES = 40
DEFAULT_RRF_K = 60
DEFAULT_RERANK_CANDIDATES = 25
DEFAULT_HYDE_WEIGHT = DEFAULT_HYDE_VARIANT_WEIGHT
CHROMA_DIR = Path("data/chroma")

# CodeRankEmbed bi-encoder: queries need this prefix
QUERY_PREFIX = "Represent this query for searching relevant code: "


def _enrich_query(query: str) -> str:
    enriched = query
    if "symfony" not in query.lower():
        enriched = f"{query} (Symfony PHP framework)"
    return enriched


def embed_queries(client: OpenAI, queries: list[str], model: str = "CodeRankEmbed") -> list[list[float]]:
    """Embed queries with the required task prefix."""
    if not queries:
        return []
    prefixed = [f"{QUERY_PREFIX}{_enrich_query(q)}" for q in queries]
    response = client.embeddings.create(input=prefixed, model=model)
    return [item.embedding for item in response.data]


def embed_query(client: OpenAI, query: str, model: str = "CodeRankEmbed") -> list[float]:
    embeddings = embed_queries(client, [query], model=model)
    return embeddings[0]


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


def retrieve_with_fusion_and_rerank(
    *,
    collection,
    bm25_index: BM25Index,
    id_to_row: dict[str, int],
    corpus_ids: list[str],
    corpus_docs: list[str],
    corpus_metas: Sequence[Any],
    query_embeddings: list[list[float]],
    vector_weights: list[float],
    query: str,
    top_k: int,
    vector_candidates: int,
    bm25_candidates: int,
    rrf_k: int,
    reranker: BGEReranker | None,
    rerank_candidates: int,
) -> dict:
    if not query_embeddings:
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    branch_ids: list[list[str]] = []
    branch_weights: list[float] = []
    vector_by_id: dict[str, tuple[str, dict, float]] = {}
    for embedding, weight in zip(query_embeddings, vector_weights):
        if weight <= 0:
            continue
        vector_results = collection.query(
            query_embeddings=[embedding],
            n_results=max(top_k, vector_candidates),
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

        branch_ids.append(vector_ids)
        branch_weights.append(float(weight))
        for doc_id, doc, meta, dist in zip(vector_ids, vector_docs, vector_metas, vector_distances):
            dist_val = float(dist) if dist is not None else 1.0
            if doc_id not in vector_by_id or dist_val < vector_by_id[doc_id][2]:
                vector_by_id[doc_id] = (str(doc or ""), dict(meta or {}), dist_val)

    bm25_hits = bm25_index.search(query, bm25_candidates)
    bm25_ids = [corpus_ids[h.index] for h in bm25_hits if 0 <= h.index < len(corpus_ids)]

    fused = reciprocal_rank_fusion(branch_ids + [bm25_ids], rrf_k=rrf_k, weights=branch_weights + [1.0])
    candidate_cap = max(top_k, rerank_candidates) if reranker else top_k
    candidate_ids = [doc_id for doc_id, _ in fused[:candidate_cap]]

    candidates: list[dict] = []
    for doc_id in candidate_ids:
        if doc_id in vector_by_id:
            doc, meta, dist = vector_by_id[doc_id]
        else:
            idx = id_to_row[doc_id]
            meta = dict(corpus_metas[idx] or {})
            doc = str(corpus_docs[idx] or "")
            dist = 0.99
        candidates.append(
            {
                "id": doc_id,
                "doc": doc,
                "meta": meta,
                "distance": dist,
            }
        )

    if reranker and candidates:
        passages = [build_rerank_passage(c["meta"], c["doc"]) for c in candidates]
        scores = reranker.score(query, passages)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = score
            c["distance"] = 1.0 - score
        candidates.sort(key=lambda c: float(c.get("rerank_score", 0.0)), reverse=True)

    final = candidates[:top_k]
    return {
        "ids": [[str(c["id"]) for c in final]],
        "documents": [[str(c["doc"]) for c in final]],
        "metadatas": [[dict(c["meta"]) for c in final]],
        "distances": [[float(c["distance"]) for c in final]],
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieve Symfony docs by similarity")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Embeddings API base URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    parser.add_argument("--vector-candidates", type=int, default=DEFAULT_VECTOR_CANDIDATES, help="Vector shortlist size before RRF")
    parser.add_argument("--bm25-candidates", type=int, default=DEFAULT_BM25_CANDIDATES, help="BM25 shortlist size before RRF")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant")
    parser.add_argument("--no-rerank", action="store_true", help="Disable BGE reranking")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL, help="BGE reranker model")
    parser.add_argument("--reranker-device", default=DEFAULT_RERANKER_DEVICE, help="Reranker device, e.g. cpu or cuda:0")
    parser.add_argument("--reranker-fp16", action="store_true", help="Use FP16 for reranker (GPU only)")
    parser.add_argument("--rerank-candidates", type=int, default=DEFAULT_RERANK_CANDIDATES, help="RRF candidates passed to reranker")
    parser.add_argument("--hyde-variants", type=int, default=DEFAULT_HYDE_VARIANTS, help="HyDE variant count for dense retrieval (0 disables)")
    parser.add_argument("--hyde-variant-weight", type=float, default=DEFAULT_HYDE_WEIGHT, help="RRF weight applied to each HyDE vector branch")
    parser.add_argument("--hyde-base-url", default=DEFAULT_HYDE_BASE_URL, help="HyDE LLM API URL")
    parser.add_argument("--hyde-model", default=DEFAULT_HYDE_MODEL, help="HyDE LLM model name")
    parser.add_argument("--hyde-temperature", type=float, default=DEFAULT_HYDE_TEMPERATURE, help="HyDE generation temperature")
    parser.add_argument("--hyde-max-chars", type=int, default=DEFAULT_HYDE_MAX_CHARS, help="Max chars kept per HyDE synthetic chunk")
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

    reranker: BGEReranker | None = None
    if not args.no_rerank:
        console.print(
            f"[bold]Loading reranker[/] {args.reranker_model} "
            f"([dim]{args.reranker_device}[/])"
        )
        reranker = BGEReranker(
            args.reranker_model,
            device=args.reranker_device,
            use_fp16=args.reranker_fp16,
            normalize=True,
        )

    hyde: HyDEGenerator | None = None
    hyde_variant_weight = max(0.0, float(args.hyde_variant_weight))
    if args.hyde_variants > 0:
        console.print(
            f"[bold]Loading HyDE[/] {args.hyde_model} "
            f"([dim]{args.hyde_base_url}[/])"
        )
        hyde = HyDEGenerator(
            base_url=args.hyde_base_url,
            model=args.hyde_model,
            variants=args.hyde_variants,
            temperature=args.hyde_temperature,
            max_chars=args.hyde_max_chars,
        )

    # Setup embeddings client
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # One-shot mode
    if args.query:
        query = " ".join(args.query)
        query_texts = [query]
        query_weights = [1.0]
        if hyde:
            for variant in hyde.generate(query):
                query_texts.append(variant)
                query_weights.append(hyde_variant_weight)
        query_embeddings = embed_queries(client, query_texts)
        results = retrieve_with_fusion_and_rerank(
            collection=collection,
            bm25_index=bm25_index,
            id_to_row=id_to_row,
            corpus_ids=corpus_ids,
            corpus_docs=corpus_docs,
            corpus_metas=corpus_metas,
            query_embeddings=query_embeddings,
            vector_weights=query_weights,
            query=query,
            top_k=args.top_k,
            vector_candidates=args.vector_candidates,
            bm25_candidates=args.bm25_candidates,
            rrf_k=args.rrf_k,
            reranker=reranker,
            rerank_candidates=args.rerank_candidates,
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
            query_texts = [query]
            query_weights = [1.0]
            if hyde:
                for variant in hyde.generate(query):
                    query_texts.append(variant)
                    query_weights.append(hyde_variant_weight)
            query_embeddings = embed_queries(client, query_texts)
            results = retrieve_with_fusion_and_rerank(
                collection=collection,
                bm25_index=bm25_index,
                id_to_row=id_to_row,
                corpus_ids=corpus_ids,
                corpus_docs=corpus_docs,
                corpus_metas=corpus_metas,
                query_embeddings=query_embeddings,
                vector_weights=query_weights,
                query=query,
                top_k=top_k,
                vector_candidates=args.vector_candidates,
                bm25_candidates=args.bm25_candidates,
                rrf_k=args.rrf_k,
                reranker=reranker,
                rerank_candidates=args.rerank_candidates,
            )
            display_results(console, query, results, top_k)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")


if __name__ == "__main__":
    main()
