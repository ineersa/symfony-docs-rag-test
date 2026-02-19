## Simple RAG - Symfony Docs

Proof-of-concept RAG pipeline for benchmarking retrieval over Symfony documentation.

### Setup

```bash
uv sync
```

### Build Index

```bash
# 1) Chunk docs
uv run python chunk.py

# 2) Embed chunks into local ChromaDB (embeddings endpoint on :8059)
uv run python embed.py --base-url http://localhost:8059/v1 --reset
```

### Retrieve Manually

```bash
# Uses dense vector + BM25 with RRF fusion, then BGE reranking
uv run python retrieve.py --base-url http://localhost:8059/v1

# Disable reranking for A/B comparison
uv run python retrieve.py --no-rerank --base-url http://localhost:8059/v1

# Enable HyDE (2 synthetic query variants)
uv run python retrieve.py --hyde-variants 2 --hyde-base-url http://localhost:4321/v1 --hyde-model local-model
```

### Build Benchmark Questions

```bash
# Fast local heuristic generation
uv run python benchmark_build.py --generator heuristic

# LLM generation (slower, model endpoint on :8052)
uv run python benchmark_build.py --generator llm --base-url http://localhost:8052/v1 --model local-model
```

Useful options:
- `--limit N` to test on a subset
- `--resume` to append while skipping duplicate IDs
- `--shard-index I --shard-count N` for split runs

Output: `data/benchmark/questions.jsonl`

### Run Benchmark

```bash
# Run both strict and relaxed scoring (local predictor uses vector+BM25+RRF+BGE rerank)
uv run python benchmark_run.py --mode both --base-url http://localhost:8059/v1

# Faster iteration: run on a random subset (e.g. 200 questions)
uv run python benchmark_run.py --mode both --base-url http://localhost:8059/v1 --sample-size 200 --sample-seed 42

# Disable reranking for local predictor
uv run python benchmark_run.py --mode both --base-url http://localhost:8059/v1 --local-no-rerank

# Enable HyDE for local simple predictor
uv run python benchmark_run.py --predictor local --mode both --local-hyde-variants 2 --local-hyde-base-url http://localhost:4321/v1 --local-hyde-model local-model
```

Notes:
- Default reranker model is `BAAI/bge-reranker-base`.
- First reranked run downloads model weights from Hugging Face.
- Override with `--reranker-model` in `retrieve.py` and `--local-reranker-model` in `benchmark_run.py`.
- HyDE is disabled by default (`--hyde-variants 0`, `--local-hyde-variants 0`).

Scoring modes:
- `strict`: source file matches and chunk line range overlaps gold line range
- `relaxed`: source file match only
- `both`: computes both

Outputs:
- Summary JSON: `data/benchmark/results.json`
- Per-question JSONL: `data/benchmark/results.jsonl`

Reported metrics include `hit@1` and `hit@5`.

### PageIndex-Style Tree RAG (Separate Pipeline)

This repo also includes an experimental, vectorless tree-retrieval pipeline inspired by PageIndex.
It is separate from the existing chunk+embedding pipeline.

```bash
# 1) Build tree index from RST docs
uv run python pageindex_build.py

# optional: generate LLM summaries during build
uv run python pageindex_build.py --summary-mode llm --summary-base-url http://localhost:8052/v1 --summary-model local-model

# 2) Query tree index (REPL or one-shot)
uv run python pageindex_retrieve.py --base-url http://localhost:8052/v1

# 3) Serve benchmark-compatible API
uv run python pageindex_api.py --base-url http://localhost:8052/v1 --port 8090

# 4) Run benchmark against PageIndex API
uv run python benchmark_run.py \
  --predictor api \
  --api-endpoint http://localhost:8090/retrieve \
  --mode both \
  --results data/benchmark/results_pageindex.json \
  --results-jsonl data/benchmark/results_pageindex.jsonl
```

Notes:
- Index artifacts are written to `data/pageindex/`.
- `pageindex_retrieve.py` and `pageindex_api.py` support `--llm-final-only` to skip tree-step LLM calls and use LLM only for final ranking.
- In LLM mode, retrieval sends both summary and body excerpts to the model (`--step-text-chars` and `--final-text-chars` tune payload size).
- Design/implementation plan is stored in `PAGEINDEX_PLAN.md`.

### Hybrid Retrieval (Summary Vectors + BM25 + Tree Expansion)

```bash
# 1) Build embeddings for PageIndex node summaries
uv run python hybrid_embed_summaries.py --base-url http://localhost:8059/v1 --reset

# 2) Query hybrid retriever (baseline)
uv run python hybrid_retrieve.py --no-llm "how to configure symfony routing"

# 3) Query hybrid retriever with HyDE (2 synthetic variants)
uv run python hybrid_retrieve.py --hyde-variants 2 "how to configure symfony routing"

# 4) Benchmark hybrid with HyDE enabled
uv run python benchmark_run.py --mode both --predictor hybrid --hybrid-hyde-variants 2
```

### Simple Web App (Separate from benchmark hybrid)

This web app keeps your existing `hybrid_retrieve.py` benchmark path untouched.
It uses:
- local llama.cpp embedding server (`coderankembed-q8_0.gguf`)
- local llama.cpp reranking server (`bge-reranker-base-q4_k_m.gguf`)
- OpenAI Chat Completions for grounded answer generation over retrieved top-5 chunks

```bash
# start local llama.cpp servers (restart=always)
docker compose -f docker-compose.llamacpp.yml up -d

# expects model files at:
# /var/models/coderankembed-q8_0.gguf
# /var/models/bge-reranker-base-q4_k_m.gguf

# required for generation
export OPENAI_API_KEY=your_key

# optional
export OPENAI_MODEL=gpt-4.1-mini
export OPENAI_BASE_URL=https://api.openai.com/v1

# optional artifact paths (defaults shown)
export WEB_NODES_FILE=data/pageindex/nodes.jsonl
export WEB_CHROMA_DIR=data/chroma
export WEB_COLLECTION=symfony_pageindex_summaries

# optional llama.cpp endpoints/model aliases (defaults shown)
export WEB_EMBED_BASE_URL=http://127.0.0.1:8059/v1
export WEB_EMBED_MODEL=CodeRankEmbed
export WEB_RERANK_BASE_URL=http://127.0.0.1:8060
export WEB_RERANK_MODEL=bge-reranker-base-q4_k_m.gguf
export WEB_RERANK_CHUNK_CHARS=700            # shorten passages for 512-token rerank models
export WEB_RERANK_CHUNK_OVERLAP=120
export WEB_LLAMACPP_API_KEY=                 # optional if llama.cpp server enables --api-key

# run web app
uv run python web_app.py
```

Run with gunicorn:

```bash
uv run gunicorn --workers 1 --threads 1 --timeout 600 --bind 127.0.0.1:8091 --capture-output --error-logfile - --access-logfile - --log-level info web_app:app
```

Stop llama.cpp services:

```bash
docker compose -f docker-compose.llamacpp.yml down
```

Run with a `.env` file:

```bash
set -a
source .env
set +a
uv run gunicorn --workers 1 --threads 1 --timeout 600 --bind 127.0.0.1:8091 --capture-output --error-logfile - --access-logfile - --log-level info web_app:app
```

Open: `http://localhost:8091`

UI includes:
- query input
- generated answer with citations (e.g. `[D1]`)
- expandable retrieved chunks with file+line in title and full chunk content inside
