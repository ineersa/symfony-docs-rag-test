#!/usr/bin/env python3
"""Embed chunked Symfony docs into ChromaDB using an OpenAI-compatible API.

Reads data/chunks.jsonl, calls the embeddings endpoint, and stores
vectors + metadata in a local ChromaDB collection.
"""

import argparse
import json
import sys
from pathlib import Path

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# ── Defaults ───────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_COLLECTION = "symfony_docs"
DEFAULT_BATCH_SIZE = 64
CHUNKS_FILE = Path("data/chunks.jsonl")
CHROMA_DIR = Path("data/chroma")


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from JSONL file."""
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def embed_batch(client: OpenAI, texts: list[str], model: str = "CodeRankEmbed") -> list[list[float]]:
    """Get embeddings for a batch of texts via OpenAI-compatible API.

    Documents are embedded as-is (no prefix) — the bi-encoder
    only needs the task prefix for queries, not documents.
    """
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def main():
    parser = argparse.ArgumentParser(description="Embed Symfony doc chunks into ChromaDB")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Embeddings API base URL")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="ChromaDB collection name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for embedding calls")
    parser.add_argument("--chunks-file", type=Path, default=CHUNKS_FILE, help="Path to chunks.jsonl")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR, help="ChromaDB storage directory")
    parser.add_argument("--reset", action="store_true", help="Delete existing collection before embedding")
    args = parser.parse_args()

    console = Console()

    # Load chunks
    if not args.chunks_file.is_file():
        console.print(f"[red]Error:[/] {args.chunks_file} not found. Run chunk.py first.")
        sys.exit(1)

    console.print(f"[bold]Loading chunks from[/] {args.chunks_file}...")
    chunks = load_chunks(args.chunks_file)
    console.print(f"Loaded [cyan]{len(chunks)}[/] chunks")

    # Setup ChromaDB
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
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    existing_count = collection.count()
    if existing_count > 0:
        console.print(f"[yellow]Collection already has {existing_count} items.[/] Use --reset to start fresh.")
        sys.exit(0)

    # Setup embeddings client
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # Embed in batches
    console.print(f"\n[bold]Embedding[/] with batch_size={args.batch_size}...")
    console.print(f"API: {args.base_url}\n")

    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding...", total=len(chunks))

        for batch_idx in range(total_batches):
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, len(chunks))
            batch = chunks[start:end]

            texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            try:
                embeddings = embed_batch(client, texts)

                # Flatten metadata values — ChromaDB requires flat string/int/float
                flat_metadatas = []
                for m in metadatas:
                    flat = {}
                    for k, v in m.items():
                        flat[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
                    flat_metadatas.append(flat)

                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=flat_metadatas,
                )
            except Exception as e:
                console.print(f"\n[red]Error on batch {batch_idx}:[/] {e}")
                console.print(f"  Chunks {start}-{end}")
                raise

            progress.update(task, advance=len(batch))

    final_count = collection.count()
    console.print(f"\n[bold green]✓[/] Stored [cyan]{final_count}[/] embeddings in ChromaDB")
    console.print(f"  Collection: {args.collection}")
    console.print(f"  Storage:    {args.chroma_dir}")


if __name__ == "__main__":
    main()
