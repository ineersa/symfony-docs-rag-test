#!/usr/bin/env python3
"""HTTP API for PageIndex/Hybrid retrieval compatible with benchmark_run.py.

Endpoint:
- POST /retrieve
  Request: {"query": "...", "top_k": 5}
  Response: {"hits": [{"id", "source", "line_start", "line_end", "distance"}]}
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hybrid_retrieve import HybridRetriever
from pageindex_common import TreeRetriever, load_nodes
from rerank_common import (
    DEFAULT_RERANK_CHUNK_CHARS,
    DEFAULT_RERANK_CHUNK_OVERLAP,
    DEFAULT_RERANKER_DEVICE,
    DEFAULT_RERANKER_MODEL,
)

DEFAULT_BASE_URL = "http://localhost:8052/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_NODES_FILE = Path("data/pageindex/nodes.jsonl")
DEFAULT_HYBRID_COLLECTION = "symfony_pageindex_summaries"
DEFAULT_HYBRID_EMBED_BASE_URL = "http://localhost:8059/v1"
DEFAULT_HYBRID_LLM_BASE_URL = "http://localhost:4321/v1"


def make_handler(retrieve_fn):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            if self.path != "/retrieve":
                self._send_json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                self._send_json(400, {"error": "invalid json"})
                return

            query = str(payload.get("query", "")).strip()
            top_k = payload.get("top_k", 5)
            try:
                top_k = int(top_k)
            except Exception:
                top_k = 5

            if not query:
                self._send_json(400, {"error": "missing query"})
                return

            try:
                hits = retrieve_fn(query, top_k)
                response = {
                    "hits": [
                        {
                            "id": h.id,
                            "source": h.source,
                            "line_start": h.line_start,
                            "line_end": h.line_end,
                            "distance": h.distance,
                        }
                        for h in hits
                    ]
                }
                self._send_json(200, response)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send_json(200, {"ok": True})
                return
            self._send_json(404, {"error": "not found"})

        def log_message(self, format, *args):  # noqa: A003
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="PageIndex benchmark-compatible retrieval API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8090, help="Bind port")
    parser.add_argument("--nodes-file", type=Path, default=DEFAULT_NODES_FILE, help="Path to pageindex nodes.jsonl")
    parser.add_argument("--mode", choices=["pageindex", "hybrid"], default="pageindex", help="Retrieval backend")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--beam-width", type=int, default=4, help="Nodes kept each depth")
    parser.add_argument("--max-depth", type=int, default=4, help="Max traversal depth")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and use lexical fallback only")
    parser.add_argument("--llm-final-only", action="store_true", help="Use only LLM for final doc selection in LLM mode")
    parser.add_argument("--step-candidates", type=int, default=24, help="Max candidates sent to LLM per traversal step")
    parser.add_argument("--final-candidates", type=int, default=24, help="Max candidates sent to LLM for final ranking")
    parser.add_argument("--step-summary-chars", type=int, default=320, help="Summary chars per candidate in traversal")
    parser.add_argument("--final-summary-chars", type=int, default=420, help="Summary chars per candidate in final ranking")
    parser.add_argument("--step-text-chars", type=int, default=900, help="Body excerpt chars per candidate in traversal")
    parser.add_argument("--final-text-chars", type=int, default=1500, help="Body excerpt chars per candidate in final ranking")
    parser.add_argument("--hybrid-collection", default=DEFAULT_HYBRID_COLLECTION, help="Chroma collection with hybrid node-summary vectors")
    parser.add_argument("--chroma-dir", type=Path, default=Path("data/chroma"), help="Chroma dir for hybrid mode")
    parser.add_argument("--hybrid-vector-top-n", type=int, default=40, help="Vector shortlist size in hybrid mode")
    parser.add_argument("--hybrid-candidate-cap", type=int, default=30, help="Candidate cap before rerank in hybrid mode")
    parser.add_argument("--hybrid-neighbor-depth", type=int, default=1, help="Neighbor expansion depth in hybrid mode")
    parser.add_argument("--hybrid-siblings-per-node", type=int, default=2, help="Sibling cap per node in hybrid mode")
    parser.add_argument("--hybrid-candidate-file-roots", type=int, default=5, help="Top file roots injected during hybrid expansion")
    parser.add_argument("--hybrid-no-expansion", action="store_true", help="Disable parent/child/sibling/file-root expansion in hybrid mode")
    parser.add_argument("--hybrid-embed-base-url", default=DEFAULT_HYBRID_EMBED_BASE_URL, help="Embedding API URL for hybrid mode")
    parser.add_argument("--hybrid-llm-base-url", default=DEFAULT_HYBRID_LLM_BASE_URL, help="LLM API URL for hybrid mode")
    parser.add_argument("--hybrid-final-char-budget", type=int, default=35000, help="Legacy option (unused); kept for backward compatibility")
    parser.add_argument("--hybrid-bm25-top-n", type=int, default=80, help="Global BM25 shortlist size before hybrid fusion")
    parser.add_argument("--hybrid-rrf-k", type=int, default=60, help="RRF constant for hybrid vector(summary)/tree + BM25 fusion")
    parser.add_argument("--hybrid-rrf-vector-weight", type=float, default=1.0, help="RRF weight for hybrid vector(summary)/tree branch")
    parser.add_argument("--hybrid-rrf-bm25-weight", type=float, default=1.0, help="RRF weight for hybrid BM25 branch")
    parser.add_argument("--hybrid-hyde-variants", type=int, default=0, help="HyDE variant count for hybrid dense retrieval (0 disables)")
    parser.add_argument("--hybrid-hyde-variant-weight", type=float, default=0.7, help="Score weight applied to each hybrid HyDE variant")
    parser.add_argument("--hybrid-hyde-temperature", type=float, default=0.3, help="Temperature for hybrid HyDE generation")
    parser.add_argument("--hybrid-hyde-max-chars", type=int, default=420, help="Max chars kept per hybrid HyDE synthetic document")
    parser.add_argument("--hybrid-no-rerank", action="store_true", help="Disable hybrid BGE rerank")
    parser.add_argument("--hybrid-reranker-model", default=DEFAULT_RERANKER_MODEL, help="BGE reranker model for hybrid mode")
    parser.add_argument("--hybrid-reranker-device", default=DEFAULT_RERANKER_DEVICE, help="Reranker device for hybrid mode")
    parser.add_argument("--hybrid-reranker-fp16", action="store_true", help="Use FP16 for hybrid reranker (GPU only)")
    parser.add_argument("--hybrid-rerank-chunk-chars", type=int, default=DEFAULT_RERANK_CHUNK_CHARS, help="Chars per rerank chunk window in hybrid mode")
    parser.add_argument("--hybrid-rerank-chunk-overlap", type=int, default=DEFAULT_RERANK_CHUNK_OVERLAP, help="Overlap chars between rerank chunks in hybrid mode")
    parser.add_argument("--hybrid-no-llm", action="store_true", help="Deprecated alias for --hybrid-no-rerank")
    args = parser.parse_args()

    if not args.nodes_file.is_file():
        raise SystemExit(f"Index file not found: {args.nodes_file}. Run pageindex_build.py first.")

    nodes = load_nodes(args.nodes_file)
    if args.mode == "pageindex":
        retriever = TreeRetriever(
            nodes,
            base_url=args.base_url,
            model=args.model,
            use_llm=not args.no_llm,
            llm_final_only=args.llm_final_only,
            step_candidate_limit=args.step_candidates,
            final_candidate_limit=args.final_candidates,
            step_summary_chars=args.step_summary_chars,
            final_summary_chars=args.final_summary_chars,
            step_text_chars=args.step_text_chars,
            final_text_chars=args.final_text_chars,
        )

        def retrieve_fn(query: str, top_k: int):
            return retriever.retrieve(query, top_k=top_k, beam_width=args.beam_width, max_depth=args.max_depth)

    else:
        retriever = HybridRetriever(
            nodes,
            base_url=args.base_url,
            embed_base_url=args.hybrid_embed_base_url,
            llm_base_url=args.hybrid_llm_base_url,
            model=args.model,
            chroma_dir=args.chroma_dir,
            collection=args.hybrid_collection,
            use_reranker=not (args.hybrid_no_rerank or args.hybrid_no_llm),
            reranker_model=args.hybrid_reranker_model,
            reranker_device=args.hybrid_reranker_device,
            reranker_fp16=args.hybrid_reranker_fp16,
            rerank_chunk_chars=args.hybrid_rerank_chunk_chars,
            rerank_chunk_overlap=args.hybrid_rerank_chunk_overlap,
            vector_top_n=args.hybrid_vector_top_n,
            candidate_cap=args.hybrid_candidate_cap,
            neighbor_depth=args.hybrid_neighbor_depth,
            siblings_per_node=args.hybrid_siblings_per_node,
            candidate_file_roots=args.hybrid_candidate_file_roots,
            enable_expansion=not args.hybrid_no_expansion,
            final_char_budget=args.hybrid_final_char_budget,
            bm25_top_n=args.hybrid_bm25_top_n,
            rrf_k=args.hybrid_rrf_k,
            rrf_vector_weight=args.hybrid_rrf_vector_weight,
            rrf_bm25_weight=args.hybrid_rrf_bm25_weight,
            hyde_variants=args.hybrid_hyde_variants,
            hyde_variant_weight=args.hybrid_hyde_variant_weight,
            hyde_temperature=args.hybrid_hyde_temperature,
            hyde_max_chars=args.hybrid_hyde_max_chars,
        )

        def retrieve_fn(query: str, top_k: int):
            return retriever.retrieve(query, top_k=top_k)

    handler_cls = make_handler(retrieve_fn)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"PageIndex API running on http://{args.host}:{args.port}")
    print("POST /retrieve  GET /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
