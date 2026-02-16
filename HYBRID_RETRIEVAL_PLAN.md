# Hybrid Retrieval Plan (Vector + BM25 + Tree)

## Updated rollout (2026-02)

1. Add BM25 with `bm25s` and use it in both pipelines:
   - simple RAG (`retrieve.py` + benchmark local predictor)
   - hybrid retriever (`hybrid_retrieve.py`)
2. Replace lexical-overlap ranking in hybrid with global BM25 branch.
3. Fuse dense and BM25 with proper Reciprocal Rank Fusion (RRF, default `k=60`).
4. Keep PageIndex-only retriever out of scope for future work.
5. Validate that commands run correctly (smoke checks only; full benchmark runs are manual).

Quick validation commands:

```bash
uv run python retrieve.py "how to configure symfony routing"
uv run python hybrid_retrieve.py --no-llm "how to configure symfony routing"
uv run python benchmark_run.py --mode both --predictor local --sample-size 20 --sample-seed 42
uv run python benchmark_run.py --mode both --predictor hybrid --sample-size 20 --sample-seed 42 --hybrid-no-llm
```

Notes:
- Use small sample sizes for verification speed; run larger benchmarks separately.
- Expose BM25/RRF tuning flags for local and hybrid predictors.

Goal: combine fast vector recall with structured PageIndex reranking to improve quality/latency over pure PageIndex traversal.

## Why Hybrid

- Vector search is fast and has strong recall.
- PageIndex/LLM is better at selecting the final, most answer-focused node.
- Full tree traversal with small models is slow and less reliable.
- Hybrid reduces LLM workload by narrowing candidates before LLM reasoning.

## Target Design

### Stage A: Vector Shortlist (Recall)

1. Embed query with current local embedding flow (`CodeRankEmbed`).
2. Retrieve top `N` chunk hits from Chroma (`N` tunable: 20/40/80).
3. Convert chunk hits to node candidates by matching:
   - `source` file
   - line overlap between chunk span and PageIndex node span
4. Build per-node evidence score from vector hits:
   - best distance
   - hit count
   - rank position weighting

Output: initial candidate node set (and candidate files).

### Stage B: Tree Expansion (Structure Context)

For each candidate node, include limited neighbors:
- parent
- children
- optional siblings (cap per node)

Also include a small number of candidate file roots inferred from Stage A.

Output: compact, structure-aware pool (target cap 20-50 nodes).

### Stage C: LLM Final Rerank (Precision)

Single LLM call (or at most 2 batched calls) to:
- rank candidates by answer likelihood,
- prefer concrete how-to sections,
- return top-k node IDs.

Fallback order if LLM fails:
1. hybrid evidence score,
2. then lexical score.

## Integration Plan

### 1) Add Hybrid Retriever Component

Create a new module (suggested: `hybrid_retrieve.py` or in `pageindex_common.py`) with:
- vector shortlist retrieval,
- chunk->node mapping,
- neighborhood expansion,
- LLM rerank.

Keep current retrievers unchanged.

### 2) Benchmark Integration

Extend `benchmark_run.py` with a new predictor:
- `--predictor hybrid`

Add flags:
- `--hybrid-base-url` (LLM/embedding endpoint if shared)
- `--hybrid-model`
- `--hybrid-vector-top-n`
- `--hybrid-candidate-cap`
- `--hybrid-neighbor-depth`
- `--hybrid-no-llm` (ablation)

### 3) Optional API Integration

Add optional endpoint mode in `pageindex_api.py` or a new `hybrid_api.py`:
- `POST /retrieve` (same response format)

## Data/Artifact Requirements

- Reuse existing:
  - `data/chunks.jsonl`
  - `data/chroma/`
  - `data/pageindex/nodes.jsonl`
- Add optional cache:
  - `data/hybrid/cache/` for query->rerank decisions.

## Evaluation Plan

Run A/B/C on the same question sample + seed:

1. Vector baseline (`--predictor local`)
2. PageIndex baseline (`--predictor pageindex`)
3. Hybrid (`--predictor hybrid`)

Report:
- strict/relaxed hit@1 and hit@5,
- latency per query (p50/p95),
- LLM calls/query,
- token usage/query (if available).

## Suggested Defaults (First Pass)

- vector top N: 40
- candidate cap to LLM: 30
- neighbor depth: 1
- siblings per node: 2
- final top-k: 5

## Risks and Mitigations

- Mapping chunk spans to nodes may miss some hits:
  - use fallback to file-level candidates.
- Candidate pool too large increases latency:
  - enforce hard cap before LLM.
- Small model rerank instability:
  - keep deterministic prompt + fallback score.

## Rollout Phases

1. Build hybrid retriever module.
2. Add `--predictor hybrid` to benchmark.
3. Run 100-question benchmark comparison.
4. Tune candidate limits and neighbor policy.
5. (Optional) expose API endpoint.
