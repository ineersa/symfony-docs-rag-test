#!/usr/bin/env python3
"""Embed PageIndex node summaries into ChromaDB for hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn


DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_COLLECTION = "symfony_pageindex_summaries"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NODES_FILE = Path("data/pageindex/nodes.jsonl")
DEFAULT_CHROMA_DIR = Path("data/chroma")


def load_nodes(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summary_embedding_text(node: dict) -> str:
    summary = str(node.get("summary") or "").strip()
    return "\n".join(
        [
            str(node.get("source") or ""),
            str(node.get("title") or ""),
            str(node.get("breadcrumb") or ""),
            summary,
        ]
    ).strip()


def embed_batch(client: OpenAI, texts: list[str], model: str = "CodeRankEmbed") -> list[list[float]]:
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed PageIndex summaries into ChromaDB")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Embeddings API base URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for embedding calls")
    parser.add_argument("--nodes-file", type=Path, default=DEFAULT_NODES_FILE, help="Path to pageindex nodes.jsonl")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR, help="ChromaDB storage directory")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection before embedding")
    args = parser.parse_args()

    console = Console()
    if not args.nodes_file.is_file():
        console.print(f"[red]Error:[/] {args.nodes_file} not found. Run pageindex_build.py first.")
        sys.exit(1)

    console.print(f"[bold]Loading nodes from[/] {args.nodes_file}...")
    nodes = load_nodes(args.nodes_file)

    corpus: list[dict] = []
    for row in nodes:
        if row.get("kind") not in {"section", "top", "file"}:
            continue
        text = summary_embedding_text(row)
        if not text:
            continue
        corpus.append({"id": str(row.get("id", "")), "text": text, "meta": row})

    if not corpus:
        console.print("[red]Error:[/] no nodes with summaries found to embed.")
        sys.exit(1)

    console.print(f"Loaded [cyan]{len(corpus)}[/] summary docs")

    args.chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(args.chroma_dir))

    if args.reset:
        try:
            chroma_client.delete_collection(args.collection)
            console.print(f"[yellow]Deleted existing collection[/] '{args.collection}'")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0:
        console.print(f"[yellow]Collection already has {existing_count} items.[/] Use --reset to start fresh.")
        sys.exit(0)

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    console.print(f"\n[bold]Embedding summaries[/] with batch_size={args.batch_size}...")
    console.print(f"API: {args.base_url}\n")

    total_batches = (len(corpus) + args.batch_size - 1) // args.batch_size
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding...", total=len(corpus))

        for batch_idx in range(total_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(corpus))
            batch = corpus[start:end]

            ids = [item["id"] for item in batch]
            texts = [item["text"] for item in batch]
            metadatas = [dict(item["meta"] or {}) for item in batch]

            try:
                embeddings = embed_batch(client, texts)

                flat_metadatas = []
                for m in metadatas:
                    flat = {}
                    for k, v in m.items():
                        flat[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
                    flat_metadatas.append(flat)

                collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=flat_metadatas)
            except Exception as exc:
                console.print(f"\n[red]Error on batch {batch_idx}:[/] {exc}")
                console.print(f"  Nodes {start}-{end}")
                raise

            progress.update(task, advance=len(batch))

    final_count = collection.count()
    console.print(f"\n[bold green]✓[/] Stored [cyan]{final_count}[/] summary embeddings in ChromaDB")
    console.print(f"  Collection: {args.collection}")
    console.print(f"  Storage:    {args.chroma_dir}")


if __name__ == "__main__":
    main()
