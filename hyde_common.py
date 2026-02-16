"""Shared HyDE helpers for simple retrieval pipelines."""

from __future__ import annotations

import json
import re

from openai import OpenAI


DEFAULT_HYDE_BASE_URL = "http://localhost:4321/v1"
DEFAULT_HYDE_MODEL = "local-model"
DEFAULT_HYDE_VARIANTS = 0
DEFAULT_HYDE_VARIANT_WEIGHT = 0.7
DEFAULT_HYDE_TEMPERATURE = 0.3
DEFAULT_HYDE_MAX_CHARS = 420
DEFAULT_HYDE_JSON_RETRIES = 2

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_obj(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_OBJ_RE.search(raw)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _clip(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


class HyDEGenerator:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        variants: int,
        temperature: float,
        max_chars: int,
    ):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.variants = max(0, variants)
        self.temperature = max(0.0, min(1.0, float(temperature)))
        self.max_chars = max(80, int(max_chars))
        self._cache: dict[str, list[str]] = {}
        self._usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def consume_usage(self) -> dict[str, int]:
        out = dict(self._usage)
        self._usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return out

    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._usage["llm_calls"] += 1
        self._usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self._usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
        self._usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)

    def generate(self, query: str) -> list[str]:
        if self.variants <= 0:
            return []

        cache_key = f"{query}|{self.variants}|{self.temperature}|{self.max_chars}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            "Generate hypothetical Symfony documentation snippets for retrieval. "
            "Write concise chunk-like passages likely to contain the answer. "
            "Use concrete config keys, command names, and how-to steps when relevant. "
            "Do not answer the user directly. "
            "Return ONLY JSON: {\"documents\": [..]}.\n\n"
            f"Query: {query}\n"
            f"Count: {self.variants}"
        )

        max_attempts = 1 + DEFAULT_HYDE_JSON_RETRIES
        docs_out: list[str] = []
        for _ in range(max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                self._record_usage(resp)
                data = _parse_json_obj(resp.choices[0].message.content or "")
                docs = data.get("documents") if isinstance(data, dict) else None
                if not isinstance(docs, list):
                    continue
                for item in docs:
                    if not isinstance(item, str):
                        continue
                    clipped = _clip(item.strip(), self.max_chars)
                    if clipped:
                        docs_out.append(clipped)
                break
            except Exception:
                continue

        deduped: list[str] = []
        seen: set[str] = set()
        for item in docs_out:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= self.variants:
                break

        self._cache[cache_key] = deduped
        return deduped
