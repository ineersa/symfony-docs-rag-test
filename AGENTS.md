# Simple RAG — Symfony Docs

Proof-of-concept RAG pipeline for benchmarking retrieval over the Symfony documentation.

## Project Structure

```
chunk.py       — RST-aware chunker (resolves includes, splits by section, preserves code blocks)
embed.py       — Embeds chunks via OpenAI-compatible API → stores in local ChromaDB
retrieve.py    — Interactive REPL for semantic search over embedded chunks
symfony-docs/  — Symfony documentation source (RST files)
data/          — Generated output (chunks.jsonl, chroma/)
```

## How to Run

```bash
# 1. Install dependencies
uv sync

# 2. Chunk the docs → data/chunks.jsonl
uv run python chunk.py

# 3. Start your llama.cpp server with CodeRankEmbed model, then embed chunks → ChromaDB
uv run python embed.py

# 4. Query the index
uv run python retrieve.py
```

## Key Details

- **Embeddings model**: `nomic-ai/CodeRankEmbed` (768-dim, 8192 context) via llama.cpp
- **Vector store**: Local ChromaDB in `data/chroma/`
- **Query handling**: Automatically prefixes queries for the bi-encoder and appends Symfony context
