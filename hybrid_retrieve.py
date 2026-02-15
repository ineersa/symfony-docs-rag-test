#!/usr/bin/env python3
"""Hybrid retrieval: vector shortlist + PageIndex reranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

from pageindex_common import QUERY_PREFIX, RetrievedHit, lexical_score, load_nodes, parse_json_obj


DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION = "symfony_docs"
DEFAULT_CHUNKS_FILE = Path("data/chunks.jsonl")
DEFAULT_NODES_FILE = Path("data/pageindex/nodes.jsonl")
DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_EMBED_BASE_URL = "http://localhost:8059/v1"
DEFAULT_LLM_BASE_URL = "http://localhost:4321/v1"
DEFAULT_MODEL = "local-model"


@dataclass
class _NodeEvidence:
    score: float = 0.0
    best_distance: float = 1.0
    hit_count: int = 0
    best_rank: int = 10_000


def _line_overlap(a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None) -> bool:
    if not isinstance(a_start, int) or not isinstance(a_end, int):
        return False
    if not isinstance(b_start, int) or not isinstance(b_end, int):
        return False
    return max(a_start, b_start) <= min(a_end, b_end)


def _line_start_key(node: dict[str, Any]) -> int:
    line_start = node.get("line_start")
    return line_start if isinstance(line_start, int) else 10**9


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
        chunks_file: Path = DEFAULT_CHUNKS_FILE,
        use_llm: bool = True,
        vector_top_n: int = 40,
        candidate_cap: int = 30,
        neighbor_depth: int = 1,
        siblings_per_node: int = 2,
        candidate_file_roots: int = 5,
        summary_chars: int = 420,
        text_chars: int = 1500,
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
        embed_url = embed_base_url or base_url
        llm_url = llm_base_url or base_url
        self.embedding_client = OpenAI(base_url=embed_url, api_key="not-needed")
        self.llm_client = OpenAI(base_url=llm_url, api_key="not-needed") if use_llm else None
        self._llm_cache: dict[str, list[str]] = {}
        self._usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = chroma_client.get_collection(collection)

        self.chunk_meta = self._load_chunk_metadata(chunks_file)
        self.nodes_by_source = self._index_nodes_by_source()
        self.file_node_by_source = self._index_file_roots()

    def consume_usage(self) -> dict[str, int]:
        out = dict(self._usage)
        self._usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return out

    def _load_chunk_metadata(self, chunks_file: Path) -> dict[str, dict]:
        by_id: dict[str, dict] = {}
        if not chunks_file.is_file():
            return by_id
        with open(chunks_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                by_id[str(row.get("id", ""))] = row.get("metadata", {})
        return by_id

    def _index_nodes_by_source(self) -> dict[str, list[dict]]:
        by_source: dict[str, list[dict]] = {}
        for n in self.nodes.values():
            source = n.get("source")
            if not source:
                continue
            if n.get("kind") not in {"section", "top"}:
                continue
            by_source.setdefault(str(source), []).append(n)
        for source, values in by_source.items():
            by_source[source] = sorted(values, key=_line_start_key)
        return by_source

    def _index_file_roots(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for n in self.nodes.values():
            if n.get("kind") == "file" and n.get("source"):
                out[str(n["source"])] = str(n["id"])
        return out

    def _embed_query(self, query: str) -> list[float]:
        enriched = query
        if "symfony" not in query.lower():
            enriched = f"{query} (Symfony PHP framework)"
        prefixed = f"{QUERY_PREFIX}{enriched}"
        response = self.embedding_client.embeddings.create(input=[prefixed], model="CodeRankEmbed")
        data = getattr(response, "data", []) or []
        if not data:
            raise RuntimeError("Embedding API returned no vectors")
        return data[0].embedding

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

    def _map_chunk_to_nodes(self, source: str, line_start: int | None, line_end: int | None, breadcrumb: str | None) -> list[str]:
        candidates = self.nodes_by_source.get(source, [])
        matched: list[str] = []
        for n in candidates:
            if _line_overlap(line_start, line_end, n.get("line_start"), n.get("line_end")):
                matched.append(str(n["id"]))
        if matched:
            return matched

        if breadcrumb:
            norm = str(breadcrumb).strip().lower()
            for n in candidates:
                if str(n.get("breadcrumb", "")).strip().lower() == norm:
                    matched.append(str(n["id"]))
        return matched

    def _vector_shortlist(self, query: str) -> tuple[dict[str, _NodeEvidence], dict[str, float]]:
        embedding = self._embed_query(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.vector_top_n,
            include=["metadatas", "distances"],
        )

        ids_raw = result.get("ids") or [[]]
        metadatas_raw = result.get("metadatas") or [[]]
        distances_raw = result.get("distances") or [[]]
        ids = ids_raw[0] if ids_raw else []
        metadatas = metadatas_raw[0] if metadatas_raw else []
        distances = distances_raw[0] if distances_raw else []

        evidence: dict[str, _NodeEvidence] = {}
        file_scores: dict[str, float] = {}

        for idx, (chunk_id, meta, dist) in enumerate(zip(ids, metadatas, distances)):
            meta = dict(meta or {})
            distance = float(dist) if dist is not None else 1.0
            rank_weight = 1.0 / (idx + 1)
            closeness = max(0.0, 1.0 - distance)
            per_hit_score = (0.7 * closeness) + (0.3 * rank_weight)

            if chunk_id in self.chunk_meta:
                merged = dict(self.chunk_meta[chunk_id])
                merged.update(meta)
                meta = merged

            source = str(meta.get("source", ""))
            if source:
                file_scores[source] = max(file_scores.get(source, 0.0), per_hit_score)

            chunk_ls = meta.get("line_start")
            chunk_le = meta.get("line_end")
            ls = int(chunk_ls) if isinstance(chunk_ls, (int, float, str)) and str(chunk_ls).isdigit() else None
            le = int(chunk_le) if isinstance(chunk_le, (int, float, str)) and str(chunk_le).isdigit() else None
            breadcrumb = str(meta.get("breadcrumb", "")) if meta.get("breadcrumb") else None

            mapped_ids = self._map_chunk_to_nodes(source, ls, le, breadcrumb)
            for node_id in mapped_ids:
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
        if not evidence and not file_scores:
            return []

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
        lexical_scores: dict[str, float] = {}

        for node_id, hybrid_score in expanded_scores.items():
            node = self.nodes.get(node_id)
            if not node:
                continue
            if node.get("kind") not in {"section", "top", "file"}:
                continue

            nodes_pool[node_id] = node
            hybrid_scores[node_id] = hybrid_score
            lexical_scores[node_id] = lexical_score(query, self._candidate_text(node))

        if not nodes_pool:
            return []

        # RRF Calculation (k=60 is standard)
        rrf_constant = 60
        final_scores: dict[str, float] = {}

        # Rank list 1: Hybrid/Vector scores
        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (nid, _) in enumerate(sorted_hybrid):
            final_scores[nid] = final_scores.get(nid, 0.0) + (1.0 / (rrf_constant + rank + 1))

        # Rank list 2: Lexical scores
        sorted_lexical = sorted(lexical_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (nid, _) in enumerate(sorted_lexical):
            final_scores[nid] = final_scores.get(nid, 0.0) + (1.0 / (rrf_constant + rank + 1))

        # Select top candidates by RRF score
        sorted_rrf = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [nodes_pool[nid] for nid, _ in sorted_rrf[: self.candidate_cap]]

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
        for idx, node in enumerate(final_nodes[:top_k]):
            if llm_ids:
                denom = max(1, top_k - 1)
                score = 1.0 if top_k == 1 else 1.0 - (idx / denom)
            else:
                score = lexical_score(query, self._candidate_text(node))
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

    parser = argparse.ArgumentParser(description="Retrieve from hybrid vector+PageIndex pipeline")
    parser.add_argument("--nodes-file", type=Path, default=DEFAULT_NODES_FILE, help="Path to pageindex nodes.jsonl")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR, help="Chroma directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection")
    parser.add_argument("--chunks-file", type=Path, default=DEFAULT_CHUNKS_FILE, help="chunks.jsonl for chunk->node mapping")
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
        chunks_file=args.chunks_file,
        use_llm=not args.no_llm,
        vector_top_n=args.vector_top_n,
        candidate_cap=args.candidate_cap,
        neighbor_depth=args.neighbor_depth,
        siblings_per_node=args.siblings_per_node,
        summary_chars=args.final_summary_chars,
        text_chars=args.final_text_chars,
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
