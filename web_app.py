#!/usr/bin/env python3
"""Simple web UI for separate hybrid retrieval + RAG generation."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
from datetime import datetime
from uuid import uuid4

from flask import Flask, render_template_string, request
from pygments.lexer import default

from pageindex_common import RetrievedHit, load_nodes
from rag_answer import NO_ANSWER_FALLBACK, RAGAnswerGenerator
from web_hybrid_retriever import (
    DEFAULT_QUERY_EMBED_REVISION,
    WebHybridRetriever,
)


APP_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Simple RAG Web</title>
  <style>
    :root {
      --bg: #f5f2ea;
      --card: #fffdf8;
      --ink: #1f2328;
      --muted: #5f6570;
      --accent: #1f6feb;
      --line: #d7d2c7;
      --ok: #1e7a45;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: radial-gradient(circle at 20% 20%, #fdf8ec, #f0ebe0 65%);
    }
    .wrap {
      max-width: 960px;
      margin: 32px auto;
      padding: 0 16px 28px;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 14px;
      box-shadow: 0 3px 10px rgba(0, 0, 0, 0.04);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.55rem;
      letter-spacing: 0.2px;
    }
    .muted { color: var(--muted); }
    form { display: grid; gap: 10px; }
    textarea {
      width: 100%;
      min-height: 110px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      font: 16px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
      resize: vertical;
      background: #fff;
    }
    .actions {
      display: flex;
      justify-content: flex-start;
      align-items: center;
      gap: 10px;
    }
    button {
      width: fit-content;
      min-height: 44px;
      border: 0;
      border-radius: 8px;
      padding: 9px 14px;
      color: white;
      background: var(--accent);
      font-weight: 600;
      cursor: pointer;
      appearance: none;
      -webkit-appearance: none;
      touch-action: manipulation;
    }
    button:disabled {
      opacity: 0.7;
      cursor: wait;
    }
    .loading-note {
      display: none;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .loading-note.visible {
      display: inline;
    }
    @media (max-width: 640px) {
      .actions {
        width: 100%;
      }
      button {
        width: 100%;
      }
    }
    .answer {
      white-space: pre-wrap;
      line-height: 1.45;
      margin-top: 8px;
      border-left: 3px solid var(--ok);
      padding-left: 10px;
    }
    details {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      margin: 8px 0;
      background: #fff;
    }
    summary {
      cursor: pointer;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
    }
    .chunk-body {
      margin-top: 8px;
      white-space: pre-wrap;
      line-height: 1.4;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Simple RAG for Symfony DOCS</h1>
      <div class="muted">
        Hybrid search with `nomic-ai/CodeRankEmbed` embeddings over summaries for tree nodes + 
        BM25 search over node texts with RRF. 
        Reranker `BAAI/bge-reranker-base` via FlagEmbedding/transformers or optional ONNX CPU backend.
        Answer generation done with Z.AI GLM-4-32B-0414-128K
      </div>
    </div>

    <div class="panel">
      <form id="search-form" method="post" action="/">
        <label for="query"><strong>Query</strong></label>
        <textarea id="query" name="query" placeholder="Ask about Symfony docs..." enterkeyhint="search">{{ query or "" }}</textarea>
        <div class="actions">
          <button id="search-btn" type="submit">Search</button>
          <span id="loading-note" class="loading-note" role="status" aria-live="polite">Searching, please wait...</span>
        </div>
      </form>
    </div>

    {% if error %}
    <div class="panel">
      <strong>Error</strong>
      <div class="answer">{{ error }}</div>
    </div>
    {% endif %}

    {% if answer %}
    <div class="panel">
      <strong>Answer</strong>
      <div class="answer">{{ answer }}</div>
    </div>
    {% endif %}

    {% if chunks %}
    <div class="panel">
      <strong>Retrieved Chunks (Top {{ chunks|length }})</strong>
      {% for c in chunks %}
      <details>
        <summary>{{ c.header }}</summary>
        <div class="chunk-body">{{ c.content }}</div>
      </details>
      {% endfor %}
    </div>
    {% endif %}
  </div>
  <script>
    (() => {
      const form = document.getElementById("search-form");
      const button = document.getElementById("search-btn");
      const loadingNote = document.getElementById("loading-note");
      if (!form || !button || !loadingNote) {
        return;
      }

      const submitForm = (event) => {
        if (event) {
          event.preventDefault();
        }
        if (form.dataset.submitting === "1") {
          return;
        }
        form.dataset.submitting = "1";
        button.disabled = true;
        button.textContent = "Searching...";
        loadingNote.classList.add("visible");
        form.submit();
      };

      button.addEventListener("click", submitForm);
      button.addEventListener("touchend", submitForm, { passive: false });
    })();
  </script>
</body>
</html>
"""


def _preview(text: str, limit: int = 120) -> str:
    one_line = " ".join((text or "").split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3].rstrip() + "..."


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    val = raw.strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _section_context_text(
    nodes: dict[str, dict],
    section_id: str,
    cache: dict[str, str],
) -> str:
    cached = cache.get(section_id)
    if cached is not None:
        return cached

    parts: list[str] = []
    visited: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visited:
            return
        visited.add(node_id)

        node = nodes.get(node_id)
        if not node or node.get("kind") not in {"section", "top"}:
            return

        text = str(node.get("text") or "")
        if text.strip():
            parts.append(text)

        for child_id in node.get("children", []):
            if not isinstance(child_id, str):
                continue
            walk(child_id)

    walk(section_id)
    combined = "\n\n".join(parts).strip()
    cache[section_id] = combined
    return combined


def _expand_hit_for_generation(
    hit: RetrievedHit,
    nodes: dict[str, dict],
    cache: dict[str, str],
) -> RetrievedHit:
    node = nodes.get(hit.id)
    if not node or node.get("kind") != "section":
        return hit

    expanded_text = _section_context_text(nodes, hit.id, cache)
    if not expanded_text or expanded_text == (hit.text or ""):
        return hit

    return RetrievedHit(
        id=hit.id,
        source=hit.source,
        line_start=hit.line_start,
        line_end=hit.line_end,
        distance=hit.distance,
        title=hit.title,
        breadcrumb=hit.breadcrumb,
        score=hit.score,
        text=expanded_text,
    )


def create_app() -> Flask:
    app = Flask(__name__)

    def log_event(message: str) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC] {message}", flush=True)

    rate_limit_seconds = 23
    ip_last_request_ts: dict[str, float] = {}
    ip_lock = threading.Lock()

    nodes_file = Path(os.getenv("WEB_NODES_FILE", "data/pageindex/nodes.jsonl"))
    chroma_dir = Path(os.getenv("WEB_CHROMA_DIR", "data/chroma"))
    chroma_collection = os.getenv("WEB_COLLECTION", "symfony_pageindex_summaries")
    query_embed_model = os.getenv("WEB_EMBED_MODEL", "nomic-ai/CodeRankEmbed")
    query_embed_revision = os.getenv("WEB_EMBED_REVISION", DEFAULT_QUERY_EMBED_REVISION)
    rerank_backend = os.getenv("WEB_RERANK_BACKEND", "onnx").strip() or "onnx"
    rerank_model = os.getenv("WEB_RERANK_MODEL", "BAAI/bge-reranker-base")
    rerank_device = os.getenv("WEB_RERANK_DEVICE", "cpu").strip() or "cpu"
    rerank_fp16 = _env_bool("WEB_RERANK_FP16", default=False)
    rerank_batch_size = _env_int("WEB_RERANK_BATCH_SIZE") or 8
    rerank_onnx_repo = os.getenv("WEB_RERANK_ONNX_REPO", "onnx-community/bge-reranker-base-ONNX")
    rerank_onnx_file = os.getenv("WEB_RERANK_ONNX_FILE", "onnx/model_int8.onnx")
    rerank_onnx_path = os.getenv("WEB_RERANK_ONNX_PATH")
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not nodes_file.is_file():
        raise RuntimeError(f"Nodes file not found: {nodes_file}")

    log_event("startup: loading pageindex nodes")
    nodes = load_nodes(nodes_file)
    log_event(f"startup: loaded {len(nodes)} nodes")
    log_event("startup: initializing retriever")
    retriever = WebHybridRetriever(
        nodes,
        chroma_dir=chroma_dir,
        collection=chroma_collection,
        query_embed_model=query_embed_model,
        query_embed_revision=query_embed_revision,
        use_reranker=True,
        reranker_backend=rerank_backend,
        reranker_model=rerank_model,
        reranker_device=rerank_device,
        reranker_fp16=rerank_fp16,
        reranker_batch_size=rerank_batch_size,
        reranker_onnx_repo=rerank_onnx_repo,
        reranker_onnx_file=rerank_onnx_file,
        reranker_onnx_path=rerank_onnx_path,
        logger=log_event,
    )
    log_event("startup: retriever ready")

    generator = None
    if openai_api_key:
        log_event("startup: initializing answer generator")
        generator = RAGAnswerGenerator(
            model=openai_model,
            api_key=openai_api_key,
            base_url=openai_base_url,
            logger=log_event,
        )
        log_event("startup: generator ready")
    else:
        log_event("startup: generator disabled (OPENAI_API_KEY missing)")

    log_event("startup: app ready")

    @app.route("/", methods=["GET", "POST"])
    def index():
        query = ""
        answer = ""
        chunks = []
        error = ""

        if request.method == "POST":
            req_id = uuid4().hex[:8]
            fwd_for = str(request.headers.get("X-Forwarded-For", "")).strip()
            client_ip = fwd_for.split(",")[0].strip() if fwd_for else (request.remote_addr or "unknown")
            now = time.time()
            log_event(f"request {req_id}: incoming from {client_ip}")
            with ip_lock:
                prev = ip_last_request_ts.get(client_ip)
                if prev is not None and (now - prev) < rate_limit_seconds:
                    retry_in = int(rate_limit_seconds - (now - prev))
                    error = f"Rate limit exceeded for {client_ip}. Retry in {retry_in}s."
                    log_event(f"request {req_id}: rate-limited, retry in {retry_in}s")
                else:
                    ip_last_request_ts[client_ip] = now

            query = str(request.form.get("query", "")).strip()
            if query and not error:
                try:
                    log_event(f"request {req_id}: retrieval start")
                    hits = retriever.retrieve(query, top_k=5)
                    log_event(f"request {req_id}: retrieval done ({len(hits)} hits)")
                    section_cache: dict[str, str] = {}
                    generation_hits = [
                        _expand_hit_for_generation(h, nodes, section_cache)
                        for h in hits
                    ]
                    if generator is not None:
                        log_event(f"request {req_id}: generation start")
                        answer, cited_chunks = generator.generate(query, generation_hits)
                        log_event(f"request {req_id}: generation done")
                    else:
                        answer = (
                            NO_ANSWER_FALLBACK
                            + "\n\n(OpenAI generation disabled because OPENAI_API_KEY is not set.)"
                        )
                        cited_chunks = []

                    src_chunks: list[dict] = []
                    if cited_chunks:
                        for c in cited_chunks:
                            src_chunks.append(
                                {
                                    "cite_id": c.cite_id,
                                    "source": c.source,
                                    "line_start": c.line_start,
                                    "line_end": c.line_end,
                                    "title": c.title,
                                    "text": c.text,
                                }
                            )
                    else:
                        for idx, h in enumerate(generation_hits[:5], start=1):
                            src_chunks.append(
                                {
                                    "cite_id": f"D{idx}",
                                    "source": h.source,
                                    "line_start": h.line_start,
                                    "line_end": h.line_end,
                                    "title": h.title,
                                    "text": h.text,
                                }
                            )

                    for c in src_chunks:
                        line = ""
                        if c.get("line_start") is not None and c.get("line_end") is not None:
                            line = f":{c['line_start']}-{c['line_end']}"
                        title = str(c.get("title") or "")
                        text = str(c.get("text") or "")
                        chunks.append(
                            {
                                "header": f"[{c.get('cite_id', 'D?')}] {c.get('source', '?')}{line} | {title} | {_preview(text)}",
                                "content": text,
                            }
                        )
                    log_event(f"request {req_id}: completed")
                except Exception as exc:
                    error = str(exc)
                    log_event(f"request {req_id}: failed: {error}")

        return render_template_string(
            APP_TEMPLATE,
            query=query,
            answer=answer,
            chunks=chunks,
            error=error,
        )

    return app

app = create_app()


if __name__ == "__main__":
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8091"))
    app.run(host=host, port=port, debug=False)
