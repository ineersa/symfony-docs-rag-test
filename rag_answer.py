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


def _clip(text: str, limit: int | None) -> str:
    clean = (text or "").strip()
    if limit is None:
        return clean
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def build_cited_chunks(hits: list[RetrievedHit], *, max_docs: int = 5, text_chars: int | None = None) -> list[CitedChunk]:
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


def _extract_citations(text: str) -> set[str]:
    cited: set[str] = set()
    for block in re.findall(r"\[([^\]]+)\]", text or ""):
        for match in re.finditer(r"(?i)\bD(\d+)\b", block):
            cited.add(f"D{match.group(1)}")
    return cited


def rewrite_followup_query(
    client: OpenAI,
    model: str,
    prev_query: str,
    prev_answer: str,
    followup: str,
    *,
    timeout: float = 30.0,
) -> str:
    """Rewrite a context-dependent follow-up into a standalone search query."""
    prompt = (
        "Rewrite the follow-up question into a standalone search query.\n"
        "Use the previous question and answer for context only.\n"
        "Output ONLY the rewritten query, nothing else.\n\n"
        f"Previous question: {prev_query}\n"
        f"Previous answer: {prev_answer[:500]}\n"
        f"Follow-up: {followup}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=timeout,
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or followup


class RAGAnswerGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        logger: Callable[[str], None] | None = None,
        request_timeout_s: float = 120.0,
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

    def _repair_citations(self, query: str, draft: str, context: str) -> str:
        repair_prompt = (
            "Rewrite the answer below to keep the same meaning but with valid citations.\n"
            "Rules:\n"
            "- Preserve factual content; do not add new facts.\n"
            "- Use only citation format [D1], [D2], etc.\n"
            "- Place citations at the end of factual sentences when possible.\n"
            "- Do not mention chunks or reasoning process.\n"
            "- If the answer is not supportable by the chunks, output exactly the fallback sentence.\n"
            f"Fallback sentence:\n{NO_ANSWER_FALLBACK}\n\n"
            f"Query:\n{query}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"Chunks:\n{context}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0,
            timeout=self._request_timeout_s,
        )
        return (resp.choices[0].message.content or "").strip()

    _SYSTEM_PROMPT = (
        "You are a Symfony documentation assistant. "
        "You are under strict grounding policy. "
        "You must use ONLY facts explicitly present in the provided chunks. "
        "No prior knowledge, no guessing, no inferred internals. "
        "Answer the user directly, never with process commentary. "
        "If evidence is missing, ambiguous, or partial, return exactly the fallback sentence. "
        "Any unsupported claim is forbidden."
    )

    def _build_user_prompt(self, query: str, context: str) -> str:
        return (
            "Use the chunks below to answer the user query.\n"
            "Hard rules (must follow all):\n"
            "1) Use only facts explicitly present in chunks.\n"
            "2) Include citations for factual statements using [D1], [D2], etc. Prefer citations at sentence ends.\n"
            "3) Do not mention chunks, context, grounding policy, or explain your reasoning process.\n"
            "4) Include concrete Symfony terms from the chunks when useful (for example attributes, class/symbol names, config keys, or short code/config snippets). Never invent terms that are not in chunks.\n"
            "5) If chunks do not directly answer the query, output exactly this sentence and nothing else:\n"
            f"{NO_ANSWER_FALLBACK}\n"
            "6) Keep answer concise and practical.\n\n"
            f"Query:\n{query}\n\n"
            f"Chunks:\n{context}"
        )

    def _validate_and_repair(
        self, query: str, final_text: str, context: str, chunks: list[CitedChunk],
    ) -> tuple[str, list[CitedChunk]]:
        """Validate citations in the generated text, attempt repair if needed."""
        if not final_text:
            return NO_ANSWER_FALLBACK, chunks
        if final_text.strip() == NO_ANSWER_FALLBACK:
            return NO_ANSWER_FALLBACK, chunks

        cited = _extract_citations(final_text)
        if not cited:
            self._log("no valid citations in draft; attempting citation repair")
            repaired = self._repair_citations(query, final_text, context)
            if repaired:
                final_text = repaired
                cited = _extract_citations(final_text)
            if not cited:
                return NO_ANSWER_FALLBACK, chunks

        allowed = {c.cite_id for c in chunks}
        if any(c not in allowed for c in cited):
            return NO_ANSWER_FALLBACK, chunks

        self._log("generation done")
        return final_text, chunks

    def generate(self, query: str, hits: list[RetrievedHit]) -> tuple[str, list[CitedChunk]]:
        self._log("building cited chunks")
        chunks = build_cited_chunks(hits, max_docs=5, text_chars=None)
        if not chunks:
            return NO_ANSWER_FALLBACK, []

        context = format_context_blocks(chunks)
        user_prompt = self._build_user_prompt(query, context)

        self._log("calling chat completion")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            timeout=self._request_timeout_s,
        )
        final_text = (resp.choices[0].message.content or "").strip()
        return self._validate_and_repair(query, final_text, context, chunks)

    def generate_followup(
        self,
        followup_query: str,
        chunks: list[CitedChunk],
        conversation_history: list[dict],
    ) -> tuple[str, list[CitedChunk]]:
        """Generate an answer for a follow-up question with conversation context.

        Args:
            followup_query: The user's follow-up question (original, not rewritten).
            chunks: Pre-built CitedChunks (merged old + new, already deduped).
            conversation_history: List of {"role": "user"|"assistant", "content": ...} dicts.
        """
        self._log("generating follow-up answer")
        if not chunks:
            return NO_ANSWER_FALLBACK, []

        context = format_context_blocks(chunks)
        user_prompt = self._build_user_prompt(followup_query, context)

        messages: list[dict] = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_prompt})

        self._log("calling chat completion (follow-up)")
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            timeout=self._request_timeout_s,
        )
        final_text = (resp.choices[0].message.content or "").strip()
        return self._validate_and_repair(followup_query, final_text, context, chunks)
