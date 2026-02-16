#!/usr/bin/env python3
"""Hybrid retrieval: summary-vector + tree expansion + BM25 + optional BGE reranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from bm25_common import BM25Index, reciprocal_rank_fusion
from pageindex_common import QUERY_PREFIX, RetrievedHit, load_nodes, parse_json_obj
from rerank_common import (
    BGEReranker,
    DEFAULT_RERANK_CHUNK_CHARS,
    DEFAULT_RERANK_CHUNK_OVERLAP,
    DEFAULT_RERANKER_DEVICE,
    DEFAULT_RERANKER_MODEL,
    split_text_for_rerank,
)


DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION = "symfony_pageindex_summaries"
DEFAULT_NODES_FILE = Path("data/pageindex/nodes.jsonl")
DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_EMBED_BASE_URL = "http://localhost:8059/v1"
DEFAULT_LLM_BASE_URL = "http://localhost:4321/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_BM25_TOP_N = 80
DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_FILE_ROOTS = 5
DEFAULT_FINAL_CHAR_BUDGET = 35_000
DEFAULT_HYDE_VARIANTS = 0
DEFAULT_HYDE_VARIANT_WEIGHT = 0.7
DEFAULT_HYDE_TEMPERATURE = 0.3
DEFAULT_HYDE_MAX_CHARS = 420
DEFAULT_HYDE_JSON_RETRIES = 2


@dataclass
class _NodeEvidence:
    score: float = 0.0
    best_distance: float = 1.0
    hit_count: int = 0
    best_rank: int = 10_000


class HybridRetriever:
    def __init__(
        self,
        nodes: dict[str, dict],
        *,
        base_url: str,
        model: str,
        embed_base_url: str | None = None,
        llm_base_url: str | None = None,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection: str = DEFAULT_COLLECTION,
        use_reranker: bool = True,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        reranker_device: str = DEFAULT_RERANKER_DEVICE,
        reranker_fp16: bool = False,
        rerank_chunk_chars: int = DEFAULT_RERANK_CHUNK_CHARS,
        rerank_chunk_overlap: int = DEFAULT_RERANK_CHUNK_OVERLAP,
        vector_top_n: int = 40,
        candidate_cap: int = 30,
        neighbor_depth: int = 1,
        siblings_per_node: int = 2,
        candidate_file_roots: int = DEFAULT_CANDIDATE_FILE_ROOTS,
        enable_expansion: bool = True,
        final_char_budget: int = DEFAULT_FINAL_CHAR_BUDGET,
        bm25_top_n: int = DEFAULT_BM25_TOP_N,
        rrf_k: int = DEFAULT_RRF_K,
        rrf_vector_weight: float = 1.0,
        rrf_bm25_weight: float = 1.0,
        hyde_variants: int = DEFAULT_HYDE_VARIANTS,
        hyde_variant_weight: float = DEFAULT_HYDE_VARIANT_WEIGHT,
        hyde_temperature: float = DEFAULT_HYDE_TEMPERATURE,
        hyde_max_chars: int = DEFAULT_HYDE_MAX_CHARS,
    ):
        self.nodes = nodes
        self.model = model
        self.use_reranker = bool(use_reranker)
        self.vector_top_n = vector_top_n
        self.candidate_cap = candidate_cap
        self.neighbor_depth = neighbor_depth
        self.siblings_per_node = siblings_per_node
        self.candidate_file_roots = candidate_file_roots
        self.enable_expansion = bool(enable_expansion)
        self.final_char_budget = max(1_000, final_char_budget)
        self.bm25_top_n = max(1, bm25_top_n)
        self.rrf_k = max(1, rrf_k)
        self.rrf_vector_weight = max(0.0, rrf_vector_weight)
        self.rrf_bm25_weight = max(0.0, rrf_bm25_weight)
        self.hyde_variants = max(0, hyde_variants)
        self.hyde_variant_weight = max(0.0, hyde_variant_weight)
        self.hyde_temperature = max(0.0, min(1.0, hyde_temperature))
        self.hyde_max_chars = max(80, hyde_max_chars)
        self.rerank_chunk_chars = max(64, int(rerank_chunk_chars))
        self.rerank_chunk_overlap = max(0, min(int(rerank_chunk_overlap), self.rerank_chunk_chars - 1))
        embed_url = embed_base_url or base_url
        llm_url = llm_base_url or base_url
        self.embedding_client = OpenAI(base_url=embed_url, api_key="not-needed")
        self.llm_client = OpenAI(base_url=llm_url, api_key="not-needed") if self.hyde_variants > 0 else None
        self.reranker = (
            BGEReranker(
                reranker_model,
                device=reranker_device,
                use_fp16=reranker_fp16,
                normalize=True,
            )
            if self.use_reranker
            else None
        )
        self._rerank_cache: dict[str, dict[str, float]] = {}
        self._hyde_cache: dict[str, list[str]] = {}
        self._usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = chroma_client.get_collection(collection)

        self.file_node_by_source = self._index_file_roots()
        self.bm25_node_ids, self.bm25 = self._build_bm25_index()

    def consume_usage(self) -> dict[str, int]:
        out = dict(self._usage)
        self._usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return out

    def _index_file_roots(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for n in self.nodes.values():
            if n.get("kind") == "file" and n.get("source"):
                out[str(n["source"])] = str(n["id"])
        return out

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."

    def _enriched_prefixed_query(self, query: str) -> str:
        enriched = query
        if "symfony" not in query.lower():
            enriched = f"{query} (Symfony PHP framework)"
        return f"{QUERY_PREFIX}{enriched}"

    def _embed_queries(self, queries: list[str]) -> list[list[float]]:
        if not queries:
            return []
        prefixed = [self._enriched_prefixed_query(q) for q in queries]
        response = self.embedding_client.embeddings.create(input=prefixed, model="CodeRankEmbed")
        data = getattr(response, "data", []) or []
        if not data:
            raise RuntimeError("Embedding API returned no vectors")
        return [item.embedding for item in data]

    def _hyde_generate_variants(self, query: str) -> list[str]:
        if self.hyde_variants <= 0 or not self.llm_client:
            return []

        cache_key = f"hyde|{query}|{self.hyde_variants}|{self.hyde_temperature}|{self.hyde_max_chars}"
        if cache_key in self._hyde_cache:
            return self._hyde_cache[cache_key]

        prompt = (
            "Generate hypothetical documentation snippets for retrieval. "
            "Write concise pseudo-document summaries likely to contain the answer. "
            "Focus on Symfony docs style (headings, config keys, command names, concrete steps). "
            "Do not answer the user directly. "
            "Return ONLY JSON with shape: {\"documents\": [..]}.\n\n"
            f"Query: {query}\n"
            f"Count: {self.hyde_variants}"
        )

        out: list[str] = []
        max_attempts = 1 + DEFAULT_HYDE_JSON_RETRIES
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.hyde_temperature,
                )
                self._record_usage(resp)
                data = parse_json_obj(resp.choices[0].message.content or "")
                docs = data.get("documents") if isinstance(data, dict) else None
                if not isinstance(docs, list):
                    print(
                        "[hybrid][hyde] Invalid JSON/shape "
                        f"(attempt {attempt}/{max_attempts}); expected {{\"documents\": [...]}}"
                    )
                    continue
                for item in docs:
                    if not isinstance(item, str):
                        continue
                    text = self._clip(item.strip(), self.hyde_max_chars)
                    if text:
                        out.append(text)
                break
            except Exception as exc:
                print(f"[hybrid][hyde] LLM call failed (attempt {attempt}/{max_attempts}): {exc}")

        deduped: list[str] = []
        seen: set[str] = set()
        for text in out:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(text)
            if len(deduped) >= self.hyde_variants:
                break

        if deduped:
            self._hyde_cache[cache_key] = deduped
        return deduped

    def _candidate_text(self, node: dict) -> str:
        return "\n".join(
            [
                node.get("source") or "",
                node.get("title") or "",
                node.get("breadcrumb") or "",
                node.get("summary") or "",
                (node.get("text") or "")[:1500],
            ]
        ).strip()

    def _node_bm25_text(self, node: dict[str, Any]) -> str:
        return "\n".join(
            [
                str(node.get("source") or ""),
                str(node.get("title") or ""),
                str(node.get("breadcrumb") or ""),
                str((node.get("text") or "")[:1800]),
            ]
        ).strip()

    def _build_bm25_index(self) -> tuple[list[str], BM25Index]:
        eligible_nodes: list[dict[str, Any]] = [
            n
            for n in self.nodes.values()
            if n.get("source") and n.get("kind") in {"section", "top", "file"}
        ]
        eligible_nodes = sorted(eligible_nodes, key=lambda n: str(n.get("id", "")))
        node_ids = [str(n["id"]) for n in eligible_nodes]
        texts = [self._node_bm25_text(n) for n in eligible_nodes]
        return node_ids, BM25Index(texts)

    def _bm25_rank(self, query: str) -> list[str]:
        hits = self.bm25.search(query, self.bm25_top_n)
        out: list[str] = []
        for h in hits:
            if 0 <= h.index < len(self.bm25_node_ids):
                out.append(self.bm25_node_ids[h.index])
        return out

    def _vector_shortlist(self, query: str) -> tuple[dict[str, _NodeEvidence], dict[str, float]]:
        query_texts = [query]
        query_weights = [1.0]
        for variant in self._hyde_generate_variants(query):
            query_texts.append(variant)
            query_weights.append(self.hyde_variant_weight)

        embeddings = self._embed_queries(query_texts)

        evidence: dict[str, _NodeEvidence] = {}
        file_scores: dict[str, float] = {}

        for embedding, weight in zip(embeddings, query_weights):
            if weight <= 0:
                continue
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=self.vector_top_n,
                include=["distances"],
            )

            ids_raw = result.get("ids") or [[]]
            distances_raw = result.get("distances") or [[]]
            ids = ids_raw[0] if ids_raw else []
            distances = distances_raw[0] if distances_raw else []

            for idx, (node_id_raw, dist) in enumerate(zip(ids, distances)):
                node_id = str(node_id_raw)
                node = self.nodes.get(node_id)
                if not node:
                    continue
                if node.get("kind") not in {"section", "top", "file"}:
                    continue

                distance = float(dist) if dist is not None else 1.0
                rank_weight = 1.0 / (idx + 1)
                closeness = max(0.0, 1.0 - distance)
                per_hit_score = ((0.7 * closeness) + (0.3 * rank_weight)) * weight

                source = str(node.get("source", ""))
                if source:
                    file_scores[source] = max(file_scores.get(source, 0.0), per_hit_score)

                e = evidence.setdefault(node_id, _NodeEvidence())
                e.score += per_hit_score
                e.hit_count += 1
                if distance < e.best_distance:
                    e.best_distance = distance
                if idx < e.best_rank:
                    e.best_rank = idx

        return evidence, file_scores

    def _expand_neighbors(self, seed_scores: dict[str, float], file_scores: dict[str, float]) -> dict[str, float]:
        if not self.enable_expansion:
            return dict(seed_scores)

        out = dict(seed_scores)
        frontier = list(seed_scores.keys())
        for _ in range(max(0, self.neighbor_depth)):
            next_frontier: list[str] = []
            for node_id in frontier:
                node = self.nodes.get(node_id)
                if not node:
                    continue
                base = out.get(node_id, 0.0)

                parent_id = node.get("parent_id")
                if isinstance(parent_id, str) and parent_id in self.nodes:
                    out[parent_id] = max(out.get(parent_id, 0.0), base * 0.75)
                    next_frontier.append(parent_id)

                for child_id in node.get("children", []):
                    if child_id in self.nodes:
                        out[child_id] = max(out.get(child_id, 0.0), base * 0.7)
                        next_frontier.append(child_id)

                if isinstance(parent_id, str) and parent_id in self.nodes:
                    siblings = [
                        sib
                        for sib in self.nodes[parent_id].get("children", [])
                        if sib != node_id and sib in self.nodes
                    ]
                    for sib_id in siblings[: self.siblings_per_node]:
                        out[sib_id] = max(out.get(sib_id, 0.0), base * 0.6)
                        next_frontier.append(sib_id)

            frontier = next_frontier

        file_sorted = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        for source, score in file_sorted[: self.candidate_file_roots]:
            file_node_id = self.file_node_by_source.get(source)
            if not file_node_id:
                continue
            out[file_node_id] = max(out.get(file_node_id, 0.0), score)
            for child_id in self.nodes.get(file_node_id, {}).get("children", [])[: self.siblings_per_node + 2]:
                if child_id in self.nodes:
                    out[child_id] = max(out.get(child_id, 0.0), score * 0.8)

        return out

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._usage["llm_calls"] += 1
        self._usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self._usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        self._usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

    def _candidate_passages(self, candidate: dict) -> list[str]:
        header = "\n".join(
            [
                str(candidate.get("source") or ""),
                str(candidate.get("title") or ""),
                str(candidate.get("breadcrumb") or ""),
                str(candidate.get("summary") or ""),
            ]
        ).strip()
        text = str(candidate.get("text") or "")
        chunks = split_text_for_rerank(
            text,
            chunk_chars=self.rerank_chunk_chars,
            overlap_chars=self.rerank_chunk_overlap,
        )
        if not chunks:
            return [header] if header else []
        out: list[str] = []
        for chunk in chunks:
            passage = "\n".join([header, chunk]).strip() if header else chunk
            if passage:
                out.append(passage)
        return out

    def _rerank_candidates(self, query: str, candidates: list[dict]) -> dict[str, float]:
        if not self.reranker or not candidates:
            return {}

        rerank_queries = [query]
        if self.hyde_variants > 0:
            rerank_queries.extend(self._hyde_generate_variants(query))
        seen_queries: set[str] = set()
        deduped_rerank_queries: list[str] = []
        for q in rerank_queries:
            key = q.strip().lower()
            if not key or key in seen_queries:
                continue
            seen_queries.add(key)
            deduped_rerank_queries.append(q)
        if not deduped_rerank_queries:
            deduped_rerank_queries = [query]

        candidate_ids = [str(c.get("id", "")) for c in candidates]
        cache_key = (
            "rerank|"
            + "|".join(deduped_rerank_queries)
            + "|"
            + "|".join(candidate_ids)
            + f"|{self.rerank_chunk_chars}|{self.rerank_chunk_overlap}"
        )
        if cache_key in self._rerank_cache:
            return dict(self._rerank_cache[cache_key])

        passages: list[str] = []
        passage_owner_ids: list[str] = []
        for candidate in candidates:
            owner_id = str(candidate.get("id", ""))
            for passage in self._candidate_passages(candidate):
                passages.append(passage)
                passage_owner_ids.append(owner_id)

        if not passages:
            return {}

        by_id: dict[str, float] = {}
        for rerank_query in deduped_rerank_queries:
            scores = self.reranker.score(rerank_query, passages)
            for owner_id, score in zip(passage_owner_ids, scores):
                prev = by_id.get(owner_id)
                if prev is None or float(score) > prev:
                    by_id[owner_id] = float(score)

        self._rerank_cache[cache_key] = dict(by_id)
        return by_id

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedHit]:
        evidence, file_scores = self._vector_shortlist(query)

        seed_ranked = sorted(
            evidence.items(),
            key=lambda x: (x[1].score, x[1].hit_count, -x[1].best_distance),
            reverse=True,
        )
        seed_scores = {node_id: ev.score for node_id, ev in seed_ranked[: self.candidate_cap]}
        expanded_scores = self._expand_neighbors(seed_scores, file_scores)

        # Collect nodes and raw scores for RRF
        nodes_pool: dict[str, dict] = {}
        hybrid_scores: dict[str, float] = {}

        for node_id, hybrid_score in expanded_scores.items():
            node = self.nodes.get(node_id)
            if not node:
                continue
            if node.get("kind") not in {"section", "top", "file"}:
                continue

            nodes_pool[node_id] = node
            hybrid_scores[node_id] = hybrid_score

        bm25_ids = self._bm25_rank(query)
        for node_id in bm25_ids:
            node = self.nodes.get(node_id)
            if not node:
                continue
            if node.get("kind") not in {"section", "top", "file"}:
                continue
            nodes_pool[node_id] = node

        if not nodes_pool:
            return []

        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_ids = [nid for nid, _ in sorted_hybrid]
        fused = reciprocal_rank_fusion(
            [hybrid_ids, bm25_ids],
            rrf_k=self.rrf_k,
            weights=[self.rrf_vector_weight, self.rrf_bm25_weight],
        )
        final_scores = {nid: score for nid, score in fused}
        candidate_ids = [nid for nid, _ in fused if nid in nodes_pool][: self.candidate_cap]
        candidates = [nodes_pool[nid] for nid in candidate_ids]

        rerank_scores = self._rerank_candidates(query, candidates) if self.use_reranker else {}
        reranked_ids: list[str] = []
        if rerank_scores:
            candidate_pos = {str(n.get("id", "")): idx for idx, n in enumerate(candidates)}
            reranked_ids = sorted(
                [str(n.get("id", "")) for n in candidates],
                key=lambda nid: (rerank_scores.get(nid, -1.0), -candidate_pos.get(nid, 0)),
                reverse=True,
            )

        final_nodes: list[dict]
        if reranked_ids:
            by_id = {n["id"]: n for n in candidates}
            final_nodes = [by_id[nid] for nid in reranked_ids if nid in by_id]
            if len(final_nodes) < top_k:
                picked = {n["id"] for n in final_nodes}
                for n in candidates:
                    if n["id"] in picked:
                        continue
                    final_nodes.append(n)
                    if len(final_nodes) >= top_k:
                        break
        else:
            final_nodes = candidates[:top_k]

        hits: list[RetrievedHit] = []
        max_fused_score = max((final_scores.get(str(n.get("id")), 0.0) for n in final_nodes), default=0.0)
        for node in final_nodes[:top_k]:
            node_id = str(node.get("id", ""))
            rerank_score = rerank_scores.get(node_id)
            if rerank_score is not None:
                score = rerank_score
            else:
                fused_score = final_scores.get(node_id, 0.0)
                score = (fused_score / max_fused_score) if max_fused_score > 0 else 0.0
            score = max(0.0, min(score, 1.0))
            hits.append(
                RetrievedHit(
                    id=str(node.get("id", "")),
                    source=str(node.get("source", "")),
                    line_start=node.get("line_start"),
                    line_end=node.get("line_end"),
                    distance=1.0 - score,
                    title=str(node.get("title", "")),
                    breadcrumb=node.get("breadcrumb"),
                    score=score,
                    text=str(node.get("text", "")),
                )
            )
        return hits


def _content_preview(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "(no extracted content)"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def display_hits(console: Console, query: str, hits: list[RetrievedHit], top_k: int, content_chars: int) -> None:
    console.print()
    console.print(Panel(f"[bold]{query}[/]", title="Query", border_style="blue"))
    if not hits:
        console.print("[yellow]No results.[/]")
        return

    for i, h in enumerate(hits, 1):
        header = f"[bold cyan]#{i}[/] [dim]score:[/] {h.score:.4f} [dim]dist:[/] {h.distance:.4f}"
        source = h.source
        if h.line_start is not None and h.line_end is not None:
            source += f":{h.line_start}-{h.line_end}"
        snippet = _content_preview(h.text, content_chars)
        body = f"[bold]{source}[/]\n[dim]{h.breadcrumb or ''}[/]\n\n{h.title}\n\n[dim]Content:[/] {snippet}"
        console.print(Panel(body, title=header, border_style="dim", padding=(0, 1)))

    console.print(f"[dim]Showing {min(len(hits), top_k)} results[/]")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve from hybrid summary-vector+PageIndex pipeline")
    parser.add_argument("--nodes-file", type=Path, default=DEFAULT_NODES_FILE, help="Path to pageindex nodes.jsonl")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR, help="Chroma directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection with node-summary vectors")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Fallback base URL for embedding/LLM calls")
    parser.add_argument("--embed-base-url", default=DEFAULT_EMBED_BASE_URL, help="Embedding API URL")
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL, help="LLM API URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name (used for HyDE if enabled)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable hybrid BGE rerank")
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL, help="BGE reranker model")
    parser.add_argument("--reranker-device", default=DEFAULT_RERANKER_DEVICE, help="Reranker device, e.g. cpu or cuda:0")
    parser.add_argument("--reranker-fp16", action="store_true", help="Use FP16 for reranker (GPU only)")
    parser.add_argument("--rerank-chunk-chars", type=int, default=DEFAULT_RERANK_CHUNK_CHARS, help="Chars per rerank chunk window")
    parser.add_argument("--rerank-chunk-overlap", type=int, default=DEFAULT_RERANK_CHUNK_OVERLAP, help="Overlap chars between rerank chunks")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--vector-top-n", type=int, default=40, help="Vector shortlist size")
    parser.add_argument("--candidate-cap", type=int, default=30, help="Max candidates before final rerank")
    parser.add_argument("--neighbor-depth", type=int, default=1, help="Neighbor expansion depth")
    parser.add_argument("--siblings-per-node", type=int, default=2, help="Sibling cap per expanded node")
    parser.add_argument("--candidate-file-roots", type=int, default=DEFAULT_CANDIDATE_FILE_ROOTS, help="Top file roots injected into expansion stage")
    parser.add_argument("--no-expansion", action="store_true", help="Disable parent/child/sibling/file-root expansion and use vector seeds only")
    parser.add_argument("--final-char-budget", type=int, default=DEFAULT_FINAL_CHAR_BUDGET, help="Legacy option (unused); kept for backward compatibility")
    parser.add_argument("--bm25-top-n", type=int, default=DEFAULT_BM25_TOP_N, help="Global BM25 shortlist size before fusion")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant for vector(summary)/tree + BM25 fusion")
    parser.add_argument("--rrf-vector-weight", type=float, default=1.0, help="RRF weight for vector(summary)/tree branch")
    parser.add_argument("--rrf-bm25-weight", type=float, default=1.0, help="RRF weight for BM25 branch")
    parser.add_argument("--hyde-variants", type=int, default=DEFAULT_HYDE_VARIANTS, help="Number of HyDE query variants for dense retrieval (0 disables)")
    parser.add_argument("--hyde-variant-weight", type=float, default=DEFAULT_HYDE_VARIANT_WEIGHT, help="Score weight applied to each HyDE variant dense hit")
    parser.add_argument("--hyde-temperature", type=float, default=DEFAULT_HYDE_TEMPERATURE, help="Temperature for HyDE generation")
    parser.add_argument("--hyde-max-chars", type=int, default=DEFAULT_HYDE_MAX_CHARS, help="Max chars kept per HyDE synthetic document")
    parser.add_argument("--content-chars", type=int, default=500, help="Max chars to show per result content")
    parser.add_argument("--no-llm", action="store_true", help="Deprecated alias for --no-rerank")
    parser.add_argument("query", nargs="*", help="One-shot query")
    args = parser.parse_args()

    console = Console()
    if not args.nodes_file.is_file():
        raise SystemExit(f"Index file not found: {args.nodes_file}. Run pageindex_build.py first.")

    nodes = load_nodes(args.nodes_file)
    retriever = HybridRetriever(
        nodes,
        base_url=args.base_url,
        embed_base_url=args.embed_base_url,
        llm_base_url=args.llm_base_url,
        model=args.model,
        chroma_dir=args.chroma_dir,
        collection=args.collection,
        use_reranker=not (args.no_rerank or args.no_llm),
        reranker_model=args.reranker_model,
        reranker_device=args.reranker_device,
        reranker_fp16=args.reranker_fp16,
        rerank_chunk_chars=args.rerank_chunk_chars,
        rerank_chunk_overlap=args.rerank_chunk_overlap,
        vector_top_n=args.vector_top_n,
        candidate_cap=args.candidate_cap,
        neighbor_depth=args.neighbor_depth,
        siblings_per_node=args.siblings_per_node,
        candidate_file_roots=args.candidate_file_roots,
        enable_expansion=not args.no_expansion,
        final_char_budget=args.final_char_budget,
        bm25_top_n=args.bm25_top_n,
        rrf_k=args.rrf_k,
        rrf_vector_weight=args.rrf_vector_weight,
        rrf_bm25_weight=args.rrf_bm25_weight,
        hyde_variants=args.hyde_variants,
        hyde_variant_weight=args.hyde_variant_weight,
        hyde_temperature=args.hyde_temperature,
        hyde_max_chars=args.hyde_max_chars,
    )

    if args.query:
        query = " ".join(args.query)
        hits = retriever.retrieve(query, top_k=args.top_k)
        display_hits(console, query, hits, args.top_k, args.content_chars)
        return

    console.print("[bold]Hybrid Retrieval REPL[/]")
    console.print("Commands: [dim]:quit, :top N, :help[/]\n")

    top_k = args.top_k
    while True:
        try:
            query = console.input("[bold green]query>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/]")
            break

        if not query:
            continue
        if query in {":quit", ":q"}:
            console.print("[dim]Bye![/]")
            break
        if query in {":help", ":h"}:
            console.print("[dim]Commands:[/]")
            console.print("  :quit / :q     - exit")
            console.print("  :top N         - set top-k")
            continue
        if query.startswith(":top "):
            try:
                top_k = int(query.split()[1])
                console.print(f"[dim]top_k={top_k}[/]")
            except Exception:
                console.print("[red]Usage: :top N[/]")
            continue

        hits = retriever.retrieve(query, top_k=top_k)
        display_hits(console, query, hits, top_k, args.content_chars)


if __name__ == "__main__":
    main()
