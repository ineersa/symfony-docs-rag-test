"""Shared BGE reranking helpers for simple retrieval pipelines."""

from __future__ import annotations

from typing import Any


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANK_CHUNK_CHARS = 1500
DEFAULT_RERANK_CHUNK_OVERLAP = 500


def build_rerank_passage(meta: dict[str, Any], doc: str) -> str:
    """Build reranker passage text from metadata + document text."""
    return "\n".join(
        [
            str(meta.get("source", "")),
            str(meta.get("breadcrumb", "")),
            str(doc or ""),
        ]
    ).strip()


class BGEReranker:
    """Thin wrapper over FlagEmbedding FlagReranker."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        device: str = DEFAULT_RERANKER_DEVICE,
        use_fp16: bool = False,
        normalize: bool = True,
        batch_size: int | None = None,
    ):
        try:
            from FlagEmbedding import FlagReranker
        except Exception as exc:
            raise RuntimeError(
                "FlagEmbedding reranker import failed. Run `uv sync` and keep `transformers<5`."
            ) from exc

        kwargs: dict[str, Any] = {"use_fp16": bool(use_fp16)}
        if device:
            kwargs["devices"] = [device]
        self._model = FlagReranker(model_name, **kwargs)
        self.normalize = normalize
        self.batch_size = max(1, int(batch_size)) if batch_size is not None else None

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return normalized relevance scores in [0,1] when supported."""
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        try:
            if self.batch_size is not None:
                scores = self._model.compute_score(
                    pairs,
                    normalize=self.normalize,
                    batch_size=self.batch_size,
                )
            else:
                scores = self._model.compute_score(pairs, normalize=self.normalize)
        except TypeError:
            scores = self._model.compute_score(pairs, normalize=self.normalize)
        if scores is None:
            return []
        if isinstance(scores, (int, float)):
            return [float(scores)]
        return [float(x) for x in scores]


def split_text_for_rerank(text: str, *, chunk_chars: int = DEFAULT_RERANK_CHUNK_CHARS, overlap_chars: int = DEFAULT_RERANK_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character windows for reranker models.

    Returns at least one chunk when text is non-empty.
    """
    clean = text or ""
    if not clean:
        return []

    size = max(64, int(chunk_chars))
    overlap = max(0, min(int(overlap_chars), size - 1))
    step = max(1, size - overlap)

    out: list[str] = []
    start = 0
    n = len(clean)
    while start < n:
        end = min(n, start + size)
        chunk = clean[start:end].strip()
        if chunk:
            out.append(chunk)
        if end >= n:
            break
        start += step

    if not out:
        return [clean[:size]]
    return out
