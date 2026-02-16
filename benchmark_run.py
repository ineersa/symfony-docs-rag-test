#!/usr/bin/env python3
"""Run retrieval benchmark over generated questions."""

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import chromadb
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from bm25_common import BM25Index, reciprocal_rank_fusion
from hybrid_retrieve import HybridRetriever
from pageindex_common import TreeRetriever, load_nodes

DEFAULT_QUESTIONS = Path("data/benchmark/questions.jsonl")
DEFAULT_RESULTS = Path("data/benchmark/results.json")
DEFAULT_RESULTS_JSONL = Path("data/benchmark/results.jsonl")

DEFAULT_BASE_URL = "http://localhost:8059/v1"
DEFAULT_PAGEINDEX_BASE_URL = "http://localhost:8052/v1"
DEFAULT_COLLECTION = "symfony_docs"
DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_TOP_K = 5
DEFAULT_LOCAL_VECTOR_CANDIDATES = 40
DEFAULT_LOCAL_BM25_CANDIDATES = 40
DEFAULT_LOCAL_RRF_K = 60
DEFAULT_PAGEINDEX_NODES = Path("data/pageindex/nodes.jsonl")
DEFAULT_PAGEINDEX_MODEL = "local-model"
DEFAULT_HYBRID_BASE_URL = DEFAULT_BASE_URL
DEFAULT_HYBRID_EMBED_BASE_URL = "http://localhost:8059/v1"
DEFAULT_HYBRID_LLM_BASE_URL = "http://localhost:4321/v1"
DEFAULT_HYBRID_COLLECTION = "symfony_pageindex_summaries"
DEFAULT_HYBRID_VECTOR_TOP_N = 40
DEFAULT_HYBRID_CANDIDATE_CAP = 30
DEFAULT_HYBRID_NEIGHBOR_DEPTH = 1
DEFAULT_HYBRID_SIBLINGS = 2

QUERY_PREFIX = "Represent this query for searching relevant code: "


@dataclass
class Hit:
    id: str
    source: str
    line_start: int | None
    line_end: int | None
    distance: float | None


class Predictor:
    def __init__(self):
        self.last_usage: dict[str, int | float] = {
            "latency_ms": 0.0,
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _set_usage(self, *, latency_ms: float, llm_calls: int = 0, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0) -> None:
        self.last_usage = {
            "latency_ms": round(latency_ms, 3),
            "llm_calls": llm_calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def predict(self, query: str, top_k: int) -> list[Hit]:
        raise NotImplementedError


class LocalChromaPredictor(Predictor):
    def __init__(
        self,
        base_url: str,
        chroma_dir: Path,
        collection: str,
        *,
        vector_candidates: int,
        bm25_candidates: int,
        rrf_k: int,
    ):
        super().__init__()
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.vector_candidates = max(1, vector_candidates)
        self.bm25_candidates = max(1, bm25_candidates)
        self.rrf_k = max(1, rrf_k)
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = chroma_client.get_collection(collection)

        corpus = self.collection.get(include=["documents", "metadatas"])
        self.corpus_ids = [str(x) for x in (corpus.get("ids") or [])]
        self.corpus_docs = list(corpus.get("documents") or [])
        self.corpus_metas = list(corpus.get("metadatas") or [])
        self.id_to_row = {doc_id: idx for idx, doc_id in enumerate(self.corpus_ids)}

        bm25_texts: list[str] = []
        for doc, meta in zip(self.corpus_docs, self.corpus_metas):
            m = dict(meta or {})
            bm25_texts.append(
                "\n".join(
                    [
                        str(m.get("source", "")),
                        str(m.get("breadcrumb", "")),
                        str(doc or ""),
                    ]
                ).strip()
            )
        self.bm25 = BM25Index(bm25_texts)

    def _embed_query(self, query: str) -> list[float]:
        enriched = query
        if "symfony" not in query.lower():
            enriched = f"{query} (Symfony PHP framework)"
        prefixed = f"{QUERY_PREFIX}{enriched}"
        response = self.client.embeddings.create(input=[prefixed], model="CodeRankEmbed")
        return response.data[0].embedding

    def predict(self, query: str, top_k: int) -> list[Hit]:
        started = time.perf_counter()
        embedding = self._embed_query(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=max(top_k, self.vector_candidates),
            include=["metadatas", "distances", "documents"],
        )

        ids_raw = result.get("ids") or [[]]
        docs_raw = result.get("documents") or [[]]
        metadatas_raw = result.get("metadatas") or [[]]
        distances_raw = result.get("distances") or [[]]
        ids = [str(x) for x in (ids_raw[0] if ids_raw else [])]
        documents = docs_raw[0] if docs_raw else []
        metadatas = metadatas_raw[0] if metadatas_raw else []
        distances = distances_raw[0] if distances_raw else []

        vector_by_id: dict[str, tuple[str, dict, float]] = {}
        for hit_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            vector_by_id[hit_id] = (str(doc or ""), dict(meta or {}), float(dist) if dist is not None else 1.0)

        bm25_hits = self.bm25.search(query, self.bm25_candidates)
        bm25_ids = [self.corpus_ids[h.index] for h in bm25_hits if 0 <= h.index < len(self.corpus_ids)]

        fused = reciprocal_rank_fusion([ids, bm25_ids], rrf_k=self.rrf_k)
        top_ids = [doc_id for doc_id, _ in fused[:top_k]]

        hits: list[Hit] = []
        for hit_id in top_ids:
            if hit_id in vector_by_id:
                _, meta, dist = vector_by_id[hit_id]
            else:
                row = self.id_to_row.get(hit_id)
                if row is None:
                    continue
                meta = dict(self.corpus_metas[row] or {})
                dist = 0.99

            ls = meta.get("line_start")
            le = meta.get("line_end")
            hits.append(
                Hit(
                    id=str(hit_id),
                    source=str(meta.get("source", "")),
                    line_start=int(ls) if isinstance(ls, (int, float, str)) and str(ls).isdigit() else None,
                    line_end=int(le) if isinstance(le, (int, float, str)) and str(le).isdigit() else None,
                    distance=float(dist) if dist is not None else None,
                )
            )
        self._set_usage(latency_ms=(time.perf_counter() - started) * 1000)
        return hits


class ApiPredictor(Predictor):
    """HTTP predictor for future external retrieval services.

    Expected response shape:
    {"hits": [{"id": "...", "source": "...", "line_start": 1, "line_end": 2, "distance": 0.1}]}
    """

    def __init__(self, endpoint: str):
        super().__init__()
        self.endpoint = endpoint

    def predict(self, query: str, top_k: int) -> list[Hit]:
        started = time.perf_counter()
        payload = json.dumps({"query": query, "top_k": top_k}).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            method="POST",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API predictor request failed: {exc}") from exc

        hits: list[Hit] = []
        for item in data.get("hits", []):
            hits.append(
                Hit(
                    id=str(item.get("id", "")),
                    source=str(item.get("source", "")),
                    line_start=item.get("line_start"),
                    line_end=item.get("line_end"),
                    distance=item.get("distance"),
                )
            )
        self._set_usage(latency_ms=(time.perf_counter() - started) * 1000)
        return hits


class LocalPageIndexPredictor(Predictor):
    def __init__(
        self,
        *,
        nodes_file: Path,
        base_url: str,
        model: str,
        no_llm: bool,
        llm_final_only: bool,
        beam_width: int,
        max_depth: int,
        step_candidates: int,
        final_candidates: int,
        step_summary_chars: int,
        final_summary_chars: int,
        step_text_chars: int,
        final_text_chars: int,
    ):
        super().__init__()
        if not nodes_file.is_file():
            raise SystemExit(f"PageIndex nodes file not found: {nodes_file}. Run pageindex_build.py first.")
        nodes = load_nodes(nodes_file)
        self.retriever = TreeRetriever(
            nodes,
            base_url=base_url,
            model=model,
            use_llm=not no_llm,
            llm_final_only=llm_final_only,
            step_candidate_limit=step_candidates,
            final_candidate_limit=final_candidates,
            step_summary_chars=step_summary_chars,
            final_summary_chars=final_summary_chars,
            step_text_chars=step_text_chars,
            final_text_chars=final_text_chars,
        )
        self.beam_width = beam_width
        self.max_depth = max_depth

    def predict(self, query: str, top_k: int) -> list[Hit]:
        started = time.perf_counter()
        raw_hits = self.retriever.retrieve(
            query,
            top_k=top_k,
            beam_width=self.beam_width,
            max_depth=self.max_depth,
        )
        out = [
            Hit(
                id=h.id,
                source=h.source,
                line_start=h.line_start,
                line_end=h.line_end,
                distance=h.distance,
            )
            for h in raw_hits
        ]
        self._set_usage(latency_ms=(time.perf_counter() - started) * 1000)
        return out


class HybridPredictor(Predictor):
    def __init__(
        self,
        *,
        nodes_file: Path,
        base_url: str,
        embed_base_url: str | None,
        llm_base_url: str | None,
        model: str,
        chroma_dir: Path,
        collection: str,
        vector_top_n: int,
        candidate_cap: int,
        neighbor_depth: int,
        siblings_per_node: int,
        no_llm: bool,
        summary_chars: int,
        text_chars: int,
        bm25_top_n: int,
        rrf_k: int,
        rrf_vector_weight: float,
        rrf_bm25_weight: float,
    ):
        super().__init__()
        if not nodes_file.is_file():
            raise SystemExit(f"Hybrid nodes file not found: {nodes_file}. Run pageindex_build.py first.")
        nodes = load_nodes(nodes_file)
        self.retriever = HybridRetriever(
            nodes,
            base_url=base_url,
            embed_base_url=embed_base_url,
            llm_base_url=llm_base_url,
            model=model,
            chroma_dir=chroma_dir,
            collection=collection,
            use_llm=not no_llm,
            vector_top_n=vector_top_n,
            candidate_cap=candidate_cap,
            neighbor_depth=neighbor_depth,
            siblings_per_node=siblings_per_node,
            summary_chars=summary_chars,
            text_chars=text_chars,
            bm25_top_n=bm25_top_n,
            rrf_k=rrf_k,
            rrf_vector_weight=rrf_vector_weight,
            rrf_bm25_weight=rrf_bm25_weight,
        )

    def predict(self, query: str, top_k: int) -> list[Hit]:
        started = time.perf_counter()
        raw_hits = self.retriever.retrieve(query, top_k=top_k)
        usage = self.retriever.consume_usage()
        self._set_usage(
            latency_ms=(time.perf_counter() - started) * 1000,
            llm_calls=int(usage.get("llm_calls", 0)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
        )
        return [
            Hit(
                id=h.id,
                source=h.source,
                line_start=h.line_start,
                line_end=h.line_end,
                distance=h.distance,
            )
            for h in raw_hits
        ]


def load_questions(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def strict_match(question: dict, hit: Hit) -> bool:
    if hit.source != question.get("source_file"):
        return False
    qs = question.get("answer_line_start")
    qe = question.get("answer_line_end")
    if not isinstance(qs, int) or not isinstance(qe, int):
        return False
    if hit.line_start is None or hit.line_end is None:
        return False
    return ranges_overlap(qs, qe, hit.line_start, hit.line_end)


def relaxed_match(question: dict, hit: Hit) -> bool:
    return hit.source == question.get("source_file")


def hit_at_k(question: dict, hits: list[Hit], k: int, mode: str) -> bool:
    subset = hits[:k]
    if mode == "strict":
        return any(strict_match(question, h) for h in subset)
    if mode == "relaxed":
        return any(relaxed_match(question, h) for h in subset)
    raise ValueError(f"Unsupported mode: {mode}")


def score_percent(numer: int, denom: int) -> float:
    if denom == 0:
        return 0.0
    return (numer / denom) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark scoring for retrieval")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="Input questions JSONL")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="Summary JSON output path")
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL, help="Per-question JSONL output path")
    parser.add_argument("--mode", choices=["strict", "relaxed", "both"], default="both", help="Scoring mode")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Maximum K to retrieve")
    parser.add_argument("--predictor", choices=["local", "api", "pageindex", "hybrid"], default="local", help="Predictor backend")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Embedding API URL for local predictor")
    parser.add_argument("--local-vector-candidates", type=int, default=DEFAULT_LOCAL_VECTOR_CANDIDATES, help="Vector shortlist size before local RRF")
    parser.add_argument("--local-bm25-candidates", type=int, default=DEFAULT_LOCAL_BM25_CANDIDATES, help="BM25 shortlist size before local RRF")
    parser.add_argument("--local-rrf-k", type=int, default=DEFAULT_LOCAL_RRF_K, help="RRF constant for local predictor")
    parser.add_argument("--pageindex-base-url", default=DEFAULT_PAGEINDEX_BASE_URL, help="LLM API URL for PageIndex predictor")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection for local predictor")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR, help="Chroma dir for local predictor")
    parser.add_argument("--api-endpoint", default="", help="HTTP endpoint for API predictor")
    parser.add_argument("--pageindex-nodes", type=Path, default=DEFAULT_PAGEINDEX_NODES, help="Nodes JSONL for PageIndex predictor")
    parser.add_argument("--pageindex-model", default=DEFAULT_PAGEINDEX_MODEL, help="Model name for PageIndex predictor")
    parser.add_argument("--pageindex-no-llm", action="store_true", help="PageIndex lexical-only mode")
    parser.add_argument("--pageindex-llm-final-only", action="store_true", help="PageIndex final-only LLM mode")
    parser.add_argument("--pageindex-beam-width", type=int, default=4, help="Beam width for PageIndex traversal")
    parser.add_argument("--pageindex-max-depth", type=int, default=4, help="Max depth for PageIndex traversal")
    parser.add_argument("--pageindex-step-candidates", type=int, default=24, help="LLM candidates per step for PageIndex")
    parser.add_argument("--pageindex-final-candidates", type=int, default=24, help="LLM candidates for final ranking in PageIndex")
    parser.add_argument("--pageindex-step-summary-chars", type=int, default=320, help="Summary chars per candidate during tree traversal")
    parser.add_argument("--pageindex-final-summary-chars", type=int, default=420, help="Summary chars per candidate during final ranking")
    parser.add_argument("--pageindex-step-text-chars", type=int, default=900, help="Body excerpt chars per candidate during tree traversal")
    parser.add_argument("--pageindex-final-text-chars", type=int, default=1500, help="Body excerpt chars per candidate during final ranking")
    parser.add_argument("--hybrid-base-url", default=DEFAULT_HYBRID_BASE_URL, help="API URL for hybrid embedding/LLM calls")
    parser.add_argument("--hybrid-embed-base-url", default=DEFAULT_HYBRID_EMBED_BASE_URL, help="Embedding API URL for hybrid")
    parser.add_argument("--hybrid-llm-base-url", default=DEFAULT_HYBRID_LLM_BASE_URL, help="LLM API URL for hybrid")
    parser.add_argument("--hybrid-model", default=DEFAULT_PAGEINDEX_MODEL, help="Model name for hybrid LLM rerank")
    parser.add_argument("--hybrid-collection", default=DEFAULT_HYBRID_COLLECTION, help="Chroma collection with hybrid node-summary vectors")
    parser.add_argument("--hybrid-vector-top-n", type=int, default=DEFAULT_HYBRID_VECTOR_TOP_N, help="Vector shortlist size for hybrid")
    parser.add_argument("--hybrid-candidate-cap", type=int, default=DEFAULT_HYBRID_CANDIDATE_CAP, help="Max candidates kept before final hybrid rerank")
    parser.add_argument("--hybrid-neighbor-depth", type=int, default=DEFAULT_HYBRID_NEIGHBOR_DEPTH, help="Neighbor expansion depth for hybrid")
    parser.add_argument("--hybrid-siblings-per-node", type=int, default=DEFAULT_HYBRID_SIBLINGS, help="Sibling cap per expanded node in hybrid")
    parser.add_argument("--hybrid-no-llm", action="store_true", help="Disable hybrid LLM rerank and use fused rank fallback")
    parser.add_argument("--hybrid-nodes", type=Path, default=DEFAULT_PAGEINDEX_NODES, help="Nodes JSONL for hybrid retrieval")
    parser.add_argument("--hybrid-bm25-top-n", type=int, default=80, help="Global BM25 shortlist size for hybrid RRF")
    parser.add_argument("--hybrid-rrf-k", type=int, default=60, help="RRF constant for hybrid vector(summary)/tree + BM25 fusion")
    parser.add_argument("--hybrid-rrf-vector-weight", type=float, default=1.0, help="RRF weight for hybrid vector(summary)/tree branch")
    parser.add_argument("--hybrid-rrf-bm25-weight", type=float, default=1.0, help="RRF weight for hybrid BM25 branch")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of benchmark questions")
    parser.add_argument("--sample-size", type=int, default=0, help="Randomly sample N questions (0 = all)")
    parser.add_argument("--sample-seed", type=int, default=42, help="Seed for random sampling")
    args = parser.parse_args()

    console = Console()
    if not args.questions.is_file():
        raise SystemExit(f"Questions file not found: {args.questions}")

    questions = load_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]
    if args.sample_size > 0 and args.sample_size < len(questions):
        rng = random.Random(args.sample_seed)
        questions = rng.sample(questions, args.sample_size)
    if not questions:
        raise SystemExit("No benchmark questions to evaluate")

    if args.predictor == "local":
        predictor = LocalChromaPredictor(
            args.base_url,
            args.chroma_dir,
            args.collection,
            vector_candidates=args.local_vector_candidates,
            bm25_candidates=args.local_bm25_candidates,
            rrf_k=args.local_rrf_k,
        )
    elif args.predictor == "api":
        if not args.api_endpoint:
            raise SystemExit("--api-endpoint is required for --predictor api")
        predictor = ApiPredictor(args.api_endpoint)
    elif args.predictor == "pageindex":
        predictor = LocalPageIndexPredictor(
            nodes_file=args.pageindex_nodes,
            base_url=args.pageindex_base_url,
            model=args.pageindex_model,
            no_llm=args.pageindex_no_llm,
            llm_final_only=args.pageindex_llm_final_only,
            beam_width=args.pageindex_beam_width,
            max_depth=args.pageindex_max_depth,
            step_candidates=args.pageindex_step_candidates,
            final_candidates=args.pageindex_final_candidates,
            step_summary_chars=args.pageindex_step_summary_chars,
            final_summary_chars=args.pageindex_final_summary_chars,
            step_text_chars=args.pageindex_step_text_chars,
            final_text_chars=args.pageindex_final_text_chars,
        )
    else:
        predictor = HybridPredictor(
            nodes_file=args.hybrid_nodes,
            base_url=args.hybrid_base_url,
            embed_base_url=args.hybrid_embed_base_url,
            llm_base_url=args.hybrid_llm_base_url,
            model=args.hybrid_model,
            chroma_dir=args.chroma_dir,
            collection=args.hybrid_collection,
            vector_top_n=args.hybrid_vector_top_n,
            candidate_cap=args.hybrid_candidate_cap,
            neighbor_depth=args.hybrid_neighbor_depth,
            siblings_per_node=args.hybrid_siblings_per_node,
            no_llm=args.hybrid_no_llm,
            summary_chars=args.pageindex_final_summary_chars,
            text_chars=args.pageindex_final_text_chars,
            bm25_top_n=args.hybrid_bm25_top_n,
            rrf_k=args.hybrid_rrf_k,
            rrf_vector_weight=args.hybrid_rrf_vector_weight,
            rrf_bm25_weight=args.hybrid_rrf_bm25_weight,
        )

    modes = ["strict", "relaxed"] if args.mode == "both" else [args.mode]

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)

    per_question: list[dict] = []
    latency_ms_values: list[float] = []
    llm_calls_values: list[int] = []
    prompt_tokens_values: list[int] = []
    completion_tokens_values: list[int] = []
    total_tokens_values: list[int] = []
    with Progress(console=console) as progress:
        task = progress.add_task("Benchmarking...", total=len(questions))
        for q in questions:
            progress.advance(task)
            query = q["question"]
            hits = predictor.predict(query, args.top_k)

            metrics: dict[str, bool] = {}
            for mode in modes:
                metrics[f"{mode}_hit@1"] = hit_at_k(q, hits, 1, mode)
                metrics[f"{mode}_hit@5"] = hit_at_k(q, hits, min(5, args.top_k), mode)

            record = {
                "id": q.get("id"),
                "question": query,
                "source_file": q.get("source_file"),
                "answer_line_start": q.get("answer_line_start"),
                "answer_line_end": q.get("answer_line_end"),
                "kind": q.get("kind"),
                "difficulty": q.get("difficulty"),
                "metrics": metrics,
                "top_hits": [
                    {
                        "id": h.id,
                        "source": h.source,
                        "line_start": h.line_start,
                        "line_end": h.line_end,
                        "distance": h.distance,
                    }
                    for h in hits[: min(args.top_k, 5)]
                ],
                "latency_ms": predictor.last_usage.get("latency_ms"),
                "llm_calls": predictor.last_usage.get("llm_calls", 0),
                "prompt_tokens": predictor.last_usage.get("prompt_tokens", 0),
                "completion_tokens": predictor.last_usage.get("completion_tokens", 0),
                "total_tokens": predictor.last_usage.get("total_tokens", 0),
            }
            per_question.append(record)
            latency = predictor.last_usage.get("latency_ms")
            if isinstance(latency, (int, float)):
                latency_ms_values.append(float(latency))
            llm_calls_values.append(int(predictor.last_usage.get("llm_calls", 0) or 0))
            prompt_tokens_values.append(int(predictor.last_usage.get("prompt_tokens", 0) or 0))
            completion_tokens_values.append(int(predictor.last_usage.get("completion_tokens", 0) or 0))
            total_tokens_values.append(int(predictor.last_usage.get("total_tokens", 0) or 0))

    summary: dict[str, dict] = {}
    total = len(per_question)
    for mode in modes:
        h1 = sum(1 for x in per_question if x["metrics"].get(f"{mode}_hit@1"))
        h5 = sum(1 for x in per_question if x["metrics"].get(f"{mode}_hit@5"))
        summary[mode] = {
            "count": total,
            "hit@1": h1,
            "hit@1_percent": round(score_percent(h1, total), 2),
            "hit@5": h5,
            "hit@5_percent": round(score_percent(h5, total), 2),
        }

    distances = [
        h["distance"]
        for q in per_question
        for h in q["top_hits"]
        if isinstance((h.get("distance")), (int, float))
    ]

    result_payload = {
        "questions_file": str(args.questions),
        "mode": args.mode,
        "top_k": args.top_k,
        "predictor": args.predictor,
        "sample": {
            "enabled": args.sample_size > 0,
            "sample_size": args.sample_size,
            "sample_seed": args.sample_seed,
        },
        "summary": summary,
        "distance_stats": {
            "count": len(distances),
            "mean": round(statistics.mean(distances), 5) if distances else None,
            "median": round(statistics.median(distances), 5) if distances else None,
        },
        "latency_ms": {
            "count": len(latency_ms_values),
            "p50": round(statistics.median(latency_ms_values), 3) if latency_ms_values else None,
            "p95": round(statistics.quantiles(latency_ms_values, n=100)[94], 3) if len(latency_ms_values) >= 20 else (round(max(latency_ms_values), 3) if latency_ms_values else None),
            "mean": round(statistics.mean(latency_ms_values), 3) if latency_ms_values else None,
        },
        "llm_usage": {
            "calls_total": sum(llm_calls_values),
            "calls_per_query_mean": round(statistics.mean(llm_calls_values), 3) if llm_calls_values else 0.0,
            "prompt_tokens_total": sum(prompt_tokens_values),
            "completion_tokens_total": sum(completion_tokens_values),
            "total_tokens_total": sum(total_tokens_values),
            "tokens_per_query_mean": round(statistics.mean(total_tokens_values), 3) if total_tokens_values else 0.0,
        },
    }

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, ensure_ascii=False, indent=2)

    with open(args.results_jsonl, "w", encoding="utf-8") as f:
        for row in per_question:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    table = Table(title="Benchmark Scores")
    table.add_column("Mode")
    table.add_column("Hit@1")
    table.add_column("Hit@5")
    table.add_column("Count")
    for mode in modes:
        row = summary[mode]
        table.add_row(
            mode,
            f"{row['hit@1']}/{row['count']} ({row['hit@1_percent']}%)",
            f"{row['hit@5']}/{row['count']} ({row['hit@5_percent']}%)",
            str(row["count"]),
        )

    console.print(table)
    console.print(f"Summary: {args.results}")
    console.print(f"Per-question: {args.results_jsonl}")


if __name__ == "__main__":
    main()
