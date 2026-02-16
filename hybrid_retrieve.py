#!/usr/bin/env python3
"""Hybrid retrieval: summary-vector + tree expansion + BM25 + optional LLM reranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from bm25_common import BM25Index, reciprocal_rank_fusion
from pageindex_common import QUERY_PREFIX, RetrievedHit, load_nodes, parse_json_obj


DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION = "symfony_pageindex_summaries"
DEFAULT_NODES_FILE = Path("data/pageindex/nodes.jsonl")
DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_EMBED_BASE_URL = "http://localhost:8059/v1"
DEFAULT_LLM_BASE_URL = "http://localhost:4321/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_BM25_TOP_N = 80
DEFAULT_RRF_K = 60
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
        use_llm: bool = True,
        vector_top_n: int = 40,
        candidate_cap: int = 30,
        neighbor_depth: int = 1,
        siblings_per_node: int = 2,
        candidate_file_roots: int = 5,
        summary_chars: int = 420,
        text_chars: int = 1500,
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
        self.use_llm = use_llm
        self.vector_top_n = vector_top_n
        self.candidate_cap = candidate_cap
        self.neighbor_depth = neighbor_depth
        self.siblings_per_node = siblings_per_node
        self.candidate_file_roots = candidate_file_roots
        self.summary_chars = summary_chars
        self.text_chars = text_chars
        self.bm25_top_n = max(1, bm25_top_n)
        self.rrf_k = max(1, rrf_k)
        self.rrf_vector_weight = max(0.0, rrf_vector_weight)
        self.rrf_bm25_weight = max(0.0, rrf_bm25_weight)
        self.hyde_variants = max(0, hyde_variants)
        self.hyde_variant_weight = max(0.0, hyde_variant_weight)
        self.hyde_temperature = max(0.0, min(1.0, hyde_temperature))
        self.hyde_max_chars = max(80, hyde_max_chars)
        embed_url = embed_base_url or base_url
        llm_url = llm_base_url or base_url
        self.embedding_client = OpenAI(base_url=embed_url, api_key="not-needed")
        self.llm_client = OpenAI(base_url=llm_url, api_key="not-needed") if (use_llm or self.hyde_variants > 0) else None
        self._llm_cache: dict[str, list[str]] = {}
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

    def _llm_rank(self, query: str, candidates: list[dict], top_k: int) -> list[str]:
        if not self.llm_client or not candidates:
            return []

        payload = [
            {
                "id": c["id"],
                "source": c.get("source"),
                "title": c.get("title"),
                "breadcrumb": c.get("breadcrumb"),
                "summary": (c.get("summary") or "")[: self.summary_chars],
                "text_excerpt": (c.get("text") or "")[: self.text_chars],
            }
            for c in candidates
        ]
        cache_key = "hybrid|" + query + "|" + "|".join(c["id"] for c in candidates) + f"|{top_k}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        prompt = (
            "You are ranking documentation nodes for a retrieval query. "
            "Pick nodes that most directly answer the question. "
            "Prefer concrete how-to sections over high-level overview pages. "
            "Return ONLY JSON: {\"ids\": [..]} in best-first order.\n\n"
            f"Query: {query}\n"
            f"Top K: {top_k}\n"
            f"Candidates: {json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            self._record_usage(resp)
            data = parse_json_obj(resp.choices[0].message.content or "")
            ids = data.get("ids") if isinstance(data, dict) else None
            if isinstance(ids, list):
                allowed = {c["id"] for c in candidates}
                ranked = [str(x) for x in ids if str(x) in allowed][:top_k]
                self._llm_cache[cache_key] = ranked
                return ranked
        except Exception:
            return []
        return []

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

        llm_ids = self._llm_rank(query, candidates, top_k) if self.use_llm else []
        final_nodes: list[dict]
        if llm_ids:
            by_id = {n["id"]: n for n in candidates}
            final_nodes = [by_id[nid] for nid in llm_ids if nid in by_id]
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
        for idx, node in enumerate(final_nodes[:top_k]):
            if llm_ids:
                denom = max(1, top_k - 1)
                score = 1.0 if top_k == 1 else 1.0 - (idx / denom)
            else:
                fused_score = final_scores.get(str(node.get("id")), 0.0)
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--vector-top-n", type=int, default=40, help="Vector shortlist size")
    parser.add_argument("--candidate-cap", type=int, default=30, help="Max candidates before final rerank")
    parser.add_argument("--neighbor-depth", type=int, default=1, help="Neighbor expansion depth")
    parser.add_argument("--siblings-per-node", type=int, default=2, help="Sibling cap per expanded node")
    parser.add_argument("--final-summary-chars", type=int, default=420, help="Summary chars per candidate for hybrid final rerank")
    parser.add_argument("--final-text-chars", type=int, default=1500, help="Body excerpt chars per candidate for hybrid final rerank")
    parser.add_argument("--bm25-top-n", type=int, default=DEFAULT_BM25_TOP_N, help="Global BM25 shortlist size before fusion")
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K, help="RRF constant for vector(summary)/tree + BM25 fusion")
    parser.add_argument("--rrf-vector-weight", type=float, default=1.0, help="RRF weight for vector(summary)/tree branch")
    parser.add_argument("--rrf-bm25-weight", type=float, default=1.0, help="RRF weight for BM25 branch")
    parser.add_argument("--hyde-variants", type=int, default=DEFAULT_HYDE_VARIANTS, help="Number of HyDE query variants for dense retrieval (0 disables)")
    parser.add_argument("--hyde-variant-weight", type=float, default=DEFAULT_HYDE_VARIANT_WEIGHT, help="Score weight applied to each HyDE variant dense hit")
    parser.add_argument("--hyde-temperature", type=float, default=DEFAULT_HYDE_TEMPERATURE, help="Temperature for HyDE generation")
    parser.add_argument("--hyde-max-chars", type=int, default=DEFAULT_HYDE_MAX_CHARS, help="Max chars kept per HyDE synthetic document")
    parser.add_argument("--content-chars", type=int, default=500, help="Max chars to show per result content")
    parser.add_argument("--no-llm", action="store_true", help="Disable hybrid LLM rerank")
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
        use_llm=not args.no_llm,
        vector_top_n=args.vector_top_n,
        candidate_cap=args.candidate_cap,
        neighbor_depth=args.neighbor_depth,
        siblings_per_node=args.siblings_per_node,
        summary_chars=args.final_summary_chars,
        text_chars=args.final_text_chars,
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
