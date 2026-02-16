"""Shared BM25 + RRF helpers for retrieval pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import bm25s


@dataclass
class BM25Hit:
    index: int
    score: float


class BM25Index:
    def __init__(self, texts: list[str]):
        self._size = len(texts)
        self._retriever = bm25s.BM25()
        if self._size == 0:
            return
        tokenized = bm25s.tokenize(texts, show_progress=False, leave=False)
        self._retriever.index(tokenized, show_progress=False, leave_progress=False)

    @property
    def size(self) -> int:
        return self._size

    def search(self, query: str, top_k: int) -> list[BM25Hit]:
        if not query.strip() or self._size == 0 or top_k <= 0:
            return []
        query_tokens = bm25s.tokenize(query, show_progress=False, leave=False)
        doc_idxs_raw, scores_raw = self._retriever.retrieve(
            query_tokens,
            k=min(top_k, self._size),
            show_progress=False,
            leave_progress=False,
            return_as="tuple",
        )
        doc_idxs = doc_idxs_raw[0].tolist()
        scores = scores_raw[0].tolist()
        return [BM25Hit(index=int(i), score=float(s)) for i, s in zip(doc_idxs, scores)]


def reciprocal_rank_fusion(
    rank_lists: list[list[str]],
    *,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    if not rank_lists:
        return []

    effective_weights = weights or [1.0] * len(rank_lists)
    if len(effective_weights) != len(rank_lists):
        raise ValueError("weights size must match rank_lists size")

    scores: dict[str, float] = {}
    for w, ranked_ids in zip(effective_weights, rank_lists):
        if w == 0:
            continue
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (float(w) / (rrf_k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
