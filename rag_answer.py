#!/usr/bin/env python3
"""Grounded RAG answer generation over retrieved hits."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from openai import OpenAI

from pageindex_common import RetrievedHit


NO_ANSWER_FALLBACK = (
    "I could not find enough grounded information in the retrieved docs. "
    "Please retry with a more specific or rephrased query."
)


@dataclass
class CitedChunk:
    cite_id: str
    source: str
    line_start: int | None
    line_end: int | None
    title: str
    breadcrumb: str | None
    text: str


def _clip(text: str, limit: int) -> str:
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def build_cited_chunks(hits: list[RetrievedHit], *, max_docs: int = 5, text_chars: int = 2000) -> list[CitedChunk]:
    out: list[CitedChunk] = []
    for idx, h in enumerate(hits[:max_docs], start=1):
        out.append(
            CitedChunk(
                cite_id=f"D{idx}",
                source=h.source,
                line_start=h.line_start,
                line_end=h.line_end,
                title=h.title,
                breadcrumb=h.breadcrumb,
                text=_clip(h.text, text_chars),
            )
        )
    return out


def format_context_blocks(chunks: list[CitedChunk]) -> str:
    blocks: list[str] = []
    for chunk in chunks:
        line_ref = ""
        if chunk.line_start is not None and chunk.line_end is not None:
            line_ref = f":{chunk.line_start}-{chunk.line_end}"
        block = "\n".join(
            [
                f"[{chunk.cite_id}] {chunk.source}{line_ref}",
                f"Title: {chunk.title}",
                f"Breadcrumb: {chunk.breadcrumb or ''}",
                "Content:",
                chunk.text,
            ]
        )
        blocks.append(block)
    return "\n\n".join(blocks)


class RAGAnswerGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        logger: Callable[[str], None] | None = None,
        request_timeout_s: float = 45.0,
    ):
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model
        self._logger = logger
        self._request_timeout_s = max(5.0, float(request_timeout_s))

    def _log(self, message: str) -> None:
        if self._logger:
            self._logger(f"[generator] {message}")

    def generate(self, query: str, hits: list[RetrievedHit]) -> tuple[str, list[CitedChunk]]:
        self._log("building cited chunks")
        chunks = build_cited_chunks(hits, max_docs=5, text_chars=2200)
        if not chunks:
            return NO_ANSWER_FALLBACK, []

        context = format_context_blocks(chunks)
        system_prompt = (
            "You are a Symfony documentation assistant. "
            "You are under strict grounding policy. "
            "You must use ONLY facts explicitly present in the provided chunks. "
            "No prior knowledge, no guessing, no inferred internals. "
            "If evidence is missing, ambiguous, or partial, return exactly the fallback sentence. "
            "Any unsupported claim is forbidden."
        )
        user_prompt = (
            "Use the chunks below to answer the user query.\n"
            "Hard rules (must follow all):\n"
            "1) Use only facts explicitly present in chunks.\n"
            "2) Every factual sentence must end with one or more citations like [D1] or [D2][D4].\n"
            "3) Do not mention symbols/class names/compiler passes/implementation details unless those exact terms appear in cited chunks.\n"
            "4) If chunks do not directly answer the query, output exactly this sentence and nothing else:\n"
            f"{NO_ANSWER_FALLBACK}\n"
            "5) Keep answer concise.\n\n"
            f"Query:\n{query}\n\n"
            f"Chunks:\n{context}"
        )

        self._log("calling chat completion")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            timeout=self._request_timeout_s,
        )
        final_text = (resp.choices[0].message.content or "").strip()
        if not final_text:
            return NO_ANSWER_FALLBACK, chunks

        if final_text.strip() == NO_ANSWER_FALLBACK:
            return NO_ANSWER_FALLBACK, chunks

        if not re.search(r"\[(D\d+)\]", final_text):
            return NO_ANSWER_FALLBACK, chunks

        cited = set(re.findall(r"\[(D\d+)\]", final_text))
        allowed = {c.cite_id for c in chunks}
        if any(c not in allowed for c in cited):
            return NO_ANSWER_FALLBACK, chunks

        self._log("generation done")
        return final_text, chunks
