#!/usr/bin/env python3
"""Simple web UI for separate hybrid retrieval + RAG generation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from datetime import datetime
from uuid import uuid4

from flask import Flask, render_template_string, request

from pageindex_common import RetrievedHit, load_nodes
from rag_answer import (
    NO_ANSWER_FALLBACK,
    CitedChunk,
    RAGAnswerGenerator,
    build_cited_chunks,
    rewrite_followup_query,
)
from web_hybrid_retriever import (
    DEFAULT_EMBED_BASE_URL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_CHUNK_CHARS,
    DEFAULT_RERANK_CHUNK_OVERLAP,
    DEFAULT_RERANK_BASE_URL,
    DEFAULT_RERANK_MODEL,
    WebHybridRetriever,
)

CONV_TTL_SECONDS = 30 * 60  # 30 minutes


@dataclass
class ConversationState:
    """Server-side state for a multi-turn conversation."""

    history: list[dict] = field(default_factory=list)
    chunk_ids: set[str] = field(default_factory=set)
    chunks: list[CitedChunk] = field(default_factory=list)
    last_query: str = ""
    last_answer: str = ""
    last_active: float = field(default_factory=time.time)


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
      --user-bg: #e8f0fe;
      --asst-bg: #f0faf0;
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
    button, .btn-link {
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
      text-decoration: none;
      font-size: inherit;
      display: inline-flex;
      align-items: center;
    }
    .btn-link.secondary {
      background: transparent;
      color: var(--muted);
      border: 1px solid var(--line);
      font-weight: 400;
    }
    .btn-link.secondary:hover {
      background: #f5f2ea;
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
      button, .btn-link {
        width: 100%;
        justify-content: center;
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
    .history-turn {
      margin-bottom: 12px;
      padding: 10px 12px;
      border-radius: 8px;
      line-height: 1.4;
    }
    .history-turn.user {
      background: var(--user-bg);
      border-left: 3px solid var(--accent);
    }
    .history-turn.assistant {
      background: var(--asst-bg);
      border-left: 3px solid var(--ok);
    }
    .history-turn .turn-label {
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .history-turn .turn-body {
      white-space: pre-wrap;
    }
    .rewrite-note {
      color: var(--muted);
      font-size: 0.88rem;
      font-style: italic;
      margin-bottom: 8px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <h1>Simple RAG for Symfony DOCS</h1>
      <div class="muted">
        Hybrid search with local llama.cpp `CodeRankEmbed` embeddings over summaries for tree nodes +
        BM25 search over node texts with RRF.
        Reranker `bge-reranker-base-q4_k_m.gguf` via local llama.cpp reranking server.
        Answer generation done with Z.AI GLM-4-32B-0414-128K
      </div>
    </div>

    {% if history %}
    <div class="panel">
      <strong>Conversation History</strong>
      {% for turn in history %}
      <div class="history-turn {{ turn.role }}">
        <div class="turn-label">{{ turn.role }}</div>
        <div class="turn-body">{{ turn.content }}</div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    <div class="panel">
      <form id="search-form" method="post" action="/">
        <input type="hidden" name="conv_id" value="{{ conv_id or '' }}" />
        <label for="query"><strong>{{ 'Follow-up' if conv_id else 'Query' }}</strong></label>
        <textarea id="query" name="query" placeholder="{{ 'Ask a follow-up question...' if conv_id else 'Ask about Symfony docs...' }}" enterkeyhint="search"></textarea>
        <div class="actions">
          <button id="search-btn" type="submit">{{ 'Ask Follow-up' if conv_id else 'Search' }}</button>
          {% if conv_id %}
          <a class="btn-link secondary" href="/">New Conversation</a>
          {% endif %}
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

    {% if rewritten_query %}
    <div class="panel">
      <div class="rewrite-note">Searched for: "{{ rewritten_query }}"</div>
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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


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


def _format_chunk_display(src_chunks: list[dict]) -> list[dict]:
    """Build the display-friendly chunk list for the template."""
    out: list[dict] = []
    for c in src_chunks:
        line = ""
        if c.get("line_start") is not None and c.get("line_end") is not None:
            line = f":{c['line_start']}-{c['line_end']}"
        title = str(c.get("title") or "")
        text = str(c.get("text") or "")
        out.append(
            {
                "header": f"[{c.get('cite_id', 'D?')}] {c.get('source', '?')}{line} | {title} | {_preview(text)}",
                "content": text,
            }
        )
    return out


def _cited_chunks_to_dicts(chunks: list[CitedChunk]) -> list[dict]:
    return [
        {
            "cite_id": c.cite_id,
            "source": c.source,
            "line_start": c.line_start,
            "line_end": c.line_end,
            "title": c.title,
            "text": c.text,
        }
        for c in chunks
    ]


def _hits_to_dicts(hits: list[RetrievedHit], start_idx: int = 1) -> list[dict]:
    return [
        {
            "cite_id": f"D{idx}",
            "source": h.source,
            "line_start": h.line_start,
            "line_end": h.line_end,
            "title": h.title,
            "text": h.text,
        }
        for idx, h in enumerate(hits[:5], start=start_idx)
    ]


def create_app() -> Flask:
    app = Flask(__name__)

    def log_event(message: str) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts} UTC] {message}", flush=True)

    rate_limit_seconds = 10
    ip_last_request_ts: dict[str, float] = {}
    ip_lock = threading.Lock()

    # Conversation store
    conversations: dict[str, ConversationState] = {}
    conv_lock = threading.Lock()

    nodes_file = Path(os.getenv("WEB_NODES_FILE", "data/pageindex/nodes.jsonl"))
    chroma_dir = Path(os.getenv("WEB_CHROMA_DIR", "data/chroma"))
    chroma_collection = os.getenv("WEB_COLLECTION", "symfony_pageindex_summaries")
    embed_base_url = os.getenv("WEB_EMBED_BASE_URL", DEFAULT_EMBED_BASE_URL).strip() or DEFAULT_EMBED_BASE_URL
    embed_model = os.getenv("WEB_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL
    rerank_base_url = os.getenv("WEB_RERANK_BASE_URL", DEFAULT_RERANK_BASE_URL).strip() or DEFAULT_RERANK_BASE_URL
    rerank_model = os.getenv("WEB_RERANK_MODEL", DEFAULT_RERANK_MODEL).strip() or DEFAULT_RERANK_MODEL
    llama_api_key = os.getenv("WEB_LLAMACPP_API_KEY", "").strip() or "not-needed"
    rerank_chunk_chars = _env_int("WEB_RERANK_CHUNK_CHARS", DEFAULT_RERANK_CHUNK_CHARS)
    rerank_chunk_overlap = _env_int("WEB_RERANK_CHUNK_OVERLAP", DEFAULT_RERANK_CHUNK_OVERLAP)
    openai_api_key = os.getenv("OPENAI_API_KEY", "not-needed").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8052/")
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
        embed_base_url=embed_base_url,
        embed_model=embed_model,
        rerank_base_url=rerank_base_url,
        use_reranker=True,
        rerank_model=rerank_model,
        llama_api_key=llama_api_key,
        rerank_chunk_chars=rerank_chunk_chars,
        rerank_chunk_overlap=rerank_chunk_overlap,
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

    def _cleanup_conversations() -> None:
        """Remove conversations older than CONV_TTL_SECONDS."""
        now = time.time()
        expired = [
            cid for cid, cs in conversations.items()
            if (now - cs.last_active) > CONV_TTL_SECONDS
        ]
        for cid in expired:
            del conversations[cid]
        if expired:
            log_event(f"cleanup: removed {len(expired)} expired conversations")

    def _handle_first_turn(
        req_id: str, query: str, section_cache: dict[str, str],
    ) -> tuple[str, list[dict], str]:
        """Handle a first-turn (new conversation) request.

        Returns (answer, src_chunks_dicts, conv_id).
        """
        log_event(f"request {req_id}: retrieval start")
        hits = retriever.retrieve(query, top_k=5)
        log_event(f"request {req_id}: retrieval done ({len(hits)} hits)")
        generation_hits = [
            _expand_hit_for_generation(h, nodes, section_cache) for h in hits
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

        src_chunks = (
            _cited_chunks_to_dicts(cited_chunks)
            if cited_chunks
            else _hits_to_dicts(generation_hits)
        )

        # Create conversation state
        conv_id = uuid4().hex[:12]
        conv = ConversationState(
            last_query=query,
            last_answer=answer,
            last_active=time.time(),
        )
        conv.history.append({"role": "user", "content": query})
        conv.history.append({"role": "assistant", "content": answer})

        # Store chunk IDs for dedup
        for h in generation_hits:
            conv.chunk_ids.add(h.id)
        if cited_chunks:
            conv.chunks = list(cited_chunks)
        else:
            conv.chunks = build_cited_chunks(generation_hits, max_docs=5)

        with conv_lock:
            conversations[conv_id] = conv

        return answer, src_chunks, conv_id

    def _handle_followup(
        req_id: str, query: str, conv_id: str, section_cache: dict[str, str],
    ) -> tuple[str, list[dict], str, str]:
        """Handle a follow-up request.

        Returns (answer, src_chunks_dicts, conv_id, rewritten_query).
        """
        with conv_lock:
            conv = conversations.get(conv_id)
        if conv is None:
            log_event(f"request {req_id}: conv_id {conv_id} expired, treating as first turn")
            answer, src_chunks, new_conv_id = _handle_first_turn(req_id, query, section_cache)
            return answer, src_chunks, new_conv_id, ""

        # Step 1: Rewrite the follow-up into a standalone query
        rewritten_query = query
        if generator is not None:
            log_event(f"request {req_id}: rewriting follow-up query")
            rewritten_query = rewrite_followup_query(
                generator.client,
                generator.model,
                conv.last_query,
                conv.last_answer,
                query,
            )
            log_event(f"request {req_id}: rewritten to: {rewritten_query[:120]}")

        # Step 2: Retrieve with rewritten query
        log_event(f"request {req_id}: follow-up retrieval start")
        new_hits = retriever.retrieve(rewritten_query, top_k=5)
        log_event(f"request {req_id}: follow-up retrieval done ({len(new_hits)} hits)")

        # Step 3: Expand and deduplicate
        new_generation_hits = [
            _expand_hit_for_generation(h, nodes, section_cache) for h in new_hits
        ]
        unique_new_hits = [h for h in new_generation_hits if h.id not in conv.chunk_ids]

        # Step 4: Merge chunks — old chunks keep their IDs, new get continuation IDs
        next_idx = len(conv.chunks) + 1
        new_cited = [
            CitedChunk(
                cite_id=f"D{next_idx + i}",
                source=h.source,
                line_start=h.line_start,
                line_end=h.line_end,
                title=h.title,
                breadcrumb=h.breadcrumb,
                text=h.text or "",
            )
            for i, h in enumerate(unique_new_hits)
        ]
        merged_chunks = conv.chunks + new_cited
        log_event(
            f"request {req_id}: merged chunks: {len(conv.chunks)} old + "
            f"{len(new_cited)} new = {len(merged_chunks)} total"
        )

        # Step 5: Generate with conversation history
        if generator is not None:
            log_event(f"request {req_id}: follow-up generation start")
            answer, final_chunks = generator.generate_followup(
                query, merged_chunks, conv.history,
            )
            log_event(f"request {req_id}: follow-up generation done")
        else:
            answer = (
                NO_ANSWER_FALLBACK
                + "\n\n(OpenAI generation disabled because OPENAI_API_KEY is not set.)"
            )
            final_chunks = merged_chunks

        src_chunks = _cited_chunks_to_dicts(final_chunks) if final_chunks else []

        # Step 6: Update conversation state
        conv.history.append({"role": "user", "content": query})
        conv.history.append({"role": "assistant", "content": answer})
        conv.last_query = query
        conv.last_answer = answer
        conv.last_active = time.time()
        for h in new_generation_hits:
            conv.chunk_ids.add(h.id)
        conv.chunks = merged_chunks

        return answer, src_chunks, conv_id, rewritten_query if rewritten_query != query else ""

    @app.route("/", methods=["GET", "POST"])
    def index():
        query = ""
        answer = ""
        chunks: list[dict] = []
        error = ""
        conv_id = ""
        history: list[dict] = []
        rewritten_query = ""

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

            # Periodic cleanup
            with conv_lock:
                _cleanup_conversations()

            query = str(request.form.get("query", "")).strip()
            incoming_conv_id = str(request.form.get("conv_id", "")).strip()

            if query and not error:
                try:
                    section_cache: dict[str, str] = {}
                    is_followup = bool(incoming_conv_id) and incoming_conv_id in conversations

                    if is_followup:
                        log_event(f"request {req_id}: follow-up for conv {incoming_conv_id}")
                        answer, src_chunks, conv_id, rewritten_query = _handle_followup(
                            req_id, query, incoming_conv_id, section_cache,
                        )
                    else:
                        log_event(f"request {req_id}: new conversation")
                        answer, src_chunks, conv_id = _handle_first_turn(
                            req_id, query, section_cache,
                        )

                    chunks = _format_chunk_display(src_chunks)
                    log_event(f"request {req_id}: completed (conv_id={conv_id})")
                except Exception as exc:
                    error = str(exc)
                    log_event(f"request {req_id}: failed: {error}")
                    conv_id = incoming_conv_id  # preserve conv_id on error

            # Get history for display (exclude the current turn which is shown as answer)
            if conv_id:
                with conv_lock:
                    conv = conversations.get(conv_id)
                if conv and len(conv.history) > 2:
                    # Show all turns except the last Q/A pair (which is shown as answer)
                    history = conv.history[:-2]

        return render_template_string(
            APP_TEMPLATE,
            query=query if not conv_id else "",
            answer=answer,
            chunks=chunks,
            error=error,
            conv_id=conv_id,
            history=history,
            rewritten_query=rewritten_query,
        )

    return app

app = create_app()


if __name__ == "__main__":
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8091"))
    app.run(host=host, port=port, debug=False)
