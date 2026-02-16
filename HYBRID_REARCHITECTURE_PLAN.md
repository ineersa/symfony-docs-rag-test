# Hybrid Rearchitecture Plan (Summaries + BM25 Text)

## Scope

- Update only the hybrid pipeline.
- Leave simple RAG and PageIndex retriever behavior unchanged.

## Target Architecture

1. Embed PageIndex node summaries (not chunks).
2. Run dense vector retrieval over summary embeddings.
3. Run BM25 over actual node text (no summary field in BM25 corpus).
4. Keep neighbor expansion in the vector branch.
5. Fuse vector+expansion and BM25 with RRF.
6. Keep final LLM rerank unchanged.

## Implementation Steps

1. Add `hybrid_embed_summaries.py` to build a Chroma collection keyed by node id.
2. Refactor `hybrid_retrieve.py` to remove chunk-to-node mapping logic.
3. Update hybrid BM25 corpus construction to use node text content.
4. Keep existing expansion and final LLM ranking flow.
5. Update hybrid wiring in `pageindex_api.py` and `benchmark_run.py` to point to the summary-vector collection defaults.

## Notes

- Default hybrid collection name: `symfony_pageindex_summaries`.
- No benchmark runs are part of this change.
