"""Shared retrieval logic for PageIndex-style tree RAG."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

QUERY_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "what",
    "when",
    "where",
    "which",
    "how",
    "why",
    "are",
    "can",
    "you",
    "your",
    "use",
    "using",
    "used",
    "configure",
    "configuring",
}

QUERY_PREFIX = "Represent this query for searching relevant code: "


@dataclass
class RetrievedHit:
    id: str
    source: str
    line_start: int | None
    line_end: int | None
    distance: float
    title: str
    breadcrumb: str | None
    score: float
    text: str


def normalize_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for raw in TOKEN_RE.findall(text or ""):
        t = raw.lower()
        if t in STOPWORDS:
            continue
        # Very light stemming for plural/verb variants (route/routes/routing, etc.)
        if len(t) > 5 and t.endswith("ing"):
            t = t[:-3]
        elif len(t) > 4 and t.endswith("ed"):
            t = t[:-2]
        elif len(t) > 4 and t.endswith("es"):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s"):
            t = t[:-1]
        if t and t not in STOPWORDS:
            out.add(t)
    return out


def lexical_score(query: str, text: str) -> float:
    q = normalize_tokens(query)
    d = normalize_tokens(text)
    if not q or not d:
        return 0.0
    inter = len(q & d)
    # Favor query coverage over pure Jaccard to avoid short-title bias.
    recall = inter / len(q)
    precision = inter / len(d)
    return (0.8 * recall) + (0.2 * precision)


def load_nodes(path: Path) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            nodes[row["id"]] = row
    return nodes


def parse_json_obj(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = QUERY_JSON_RE.search(text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}


class TreeRetriever:
    def __init__(
        self,
        nodes: dict[str, dict],
        *,
        base_url: str,
        model: str,
        use_llm: bool = True,
        llm_final_only: bool = False,
        step_candidate_limit: int = 24,
        final_candidate_limit: int = 24,
        step_summary_chars: int = 320,
        final_summary_chars: int = 420,
        step_text_chars: int = 900,
        final_text_chars: int = 1500,
    ):
        self.nodes = nodes
        self.model = model
        self.use_llm = use_llm
        self.llm_final_only = llm_final_only
        self.step_candidate_limit = step_candidate_limit
        self.final_candidate_limit = final_candidate_limit
        self.step_summary_chars = step_summary_chars
        self.final_summary_chars = final_summary_chars
        self.step_text_chars = step_text_chars
        self.final_text_chars = final_text_chars
        self.client = OpenAI(base_url=base_url, api_key="not-needed") if use_llm else None
        self._llm_cache: dict[str, list[str]] = {}

    def _children(self, node_id: str) -> list[dict]:
        node = self.nodes[node_id]
        return [self.nodes[cid] for cid in node.get("children", []) if cid in self.nodes]

    def _candidate_text(self, node: dict) -> str:
        return "\n".join(
            [
                node.get("source") or "",
                node.get("title") or "",
                node.get("breadcrumb") or "",
                node.get("summary") or "",
            ]
        ).strip()

    def _candidate_text_with_body(self, node: dict) -> str:
        # Cap body to keep lexical scoring fast while still useful.
        body = (node.get("text") or "")[:2000]
        return (self._candidate_text(node) + "\n" + body).strip()

    @staticmethod
    def _clip(text: str, max_chars: int) -> str:
        clean = " ".join((text or "").split())
        if len(clean) <= max_chars:
            return clean
        return clean[: max_chars - 3].rstrip() + "..."

    def _llm_pick_children(self, query: str, candidates: list[dict], max_pick: int) -> list[str]:
        if not self.client or not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda c: lexical_score(query, self._candidate_text(c)),
            reverse=True,
        )
        candidates = ranked[: self.step_candidate_limit]
        cache_key = "step|" + query + "|" + "|".join(c["id"] for c in candidates) + f"|{max_pick}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        payload = [
            {
                "id": c["id"],
                "title": c.get("title"),
                "source": c.get("source"),
                "breadcrumb": c.get("breadcrumb"),
                "summary": self._clip(c.get("summary") or "", self.step_summary_chars),
                "text_excerpt": self._clip(c.get("text") or "", self.step_text_chars),
            }
            for c in candidates
        ]

        prompt = (
            "You are selecting relevant tree nodes for a documentation query. "
            "Choose the best candidate node IDs likely to contain the answer. "
            "Return ONLY JSON with shape: {\"ids\": [..]}.\n\n"
            f"Query: {query}\n"
            f"Max IDs: {max_pick}\n"
            f"Candidates: {json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            data = parse_json_obj(resp.choices[0].message.content or "")
            ids = data.get("ids") if isinstance(data, dict) else None
            if isinstance(ids, list):
                out = [str(x) for x in ids if str(x) in {c["id"] for c in candidates}]
                out = out[:max_pick]
                self._llm_cache[cache_key] = out
                return out
        except Exception:
            return []
        return []

    def _llm_rank_candidates(self, query: str, candidates: list[dict], top_k: int) -> list[str]:
        """Ask the LLM to choose and order final result nodes.

        Returns a list of node ids in ranked order.
        """
        if not self.client or not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda c: lexical_score(query, self._candidate_text_with_body(c)),
            reverse=True,
        )
        candidates = ranked[: self.final_candidate_limit]
        cache_key = "final|" + query + "|" + "|".join(c["id"] for c in candidates) + f"|{top_k}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        payload = [
            {
                "id": c["id"],
                "source": c.get("source"),
                "title": c.get("title"),
                "breadcrumb": c.get("breadcrumb"),
                "summary": self._clip(c.get("summary") or "", self.final_summary_chars),
                "text_excerpt": self._clip(c.get("text") or "", self.final_text_chars),
            }
            for c in candidates
        ]

        prompt = (
            "You are ranking documentation nodes for a retrieval query. "
            "Pick the best nodes that most directly answer the query. "
            "Prefer concrete how-to sections over generic overview pages. "
            "Return ONLY JSON: {\"ids\": [..]} in best-first order.\n\n"
            f"Query: {query}\n"
            f"Top K: {top_k}\n"
            f"Candidates: {json.dumps(payload, ensure_ascii=False)}"
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            data = parse_json_obj(resp.choices[0].message.content or "")
            ids = data.get("ids") if isinstance(data, dict) else None
            if isinstance(ids, list):
                allowed = {c["id"] for c in candidates}
                out = [str(x) for x in ids if str(x) in allowed]
                out = out[:top_k]
                self._llm_cache[cache_key] = out
                return out
        except Exception:
            return []
        return []

    def _pick_children(self, query: str, candidates: list[dict], max_pick: int) -> list[dict]:
        if not candidates:
            return []

        chosen_ids = self._llm_pick_children(query, candidates, max_pick) if (self.use_llm and not self.llm_final_only) else []
        if chosen_ids:
            id_to_node = {c["id"]: c for c in candidates}
            return [id_to_node[cid] for cid in chosen_ids if cid in id_to_node]

        ranked = sorted(
            candidates,
            key=lambda c: lexical_score(query, self._candidate_text(c)),
            reverse=True,
        )
        return ranked[:max_pick]

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        beam_width: int = 4,
        max_depth: int = 4,
    ) -> list[RetrievedHit]:
        if "pageindex_root" not in self.nodes:
            raise RuntimeError("Missing root node 'pageindex_root' in index")

        # Lexical-only mode: rank globally across content-bearing nodes.
        # This avoids tree-path lock-in and improves recall when wording differs.
        if not self.use_llm:
            pool = [
                n
                for n in self.nodes.values()
                if n.get("source") and n.get("kind") in {"section", "top"} and (n.get("text") or n.get("summary"))
            ]
            ranked = sorted(pool, key=lambda n: lexical_score(query, self._candidate_text_with_body(n)), reverse=True)
            hits: list[RetrievedHit] = []
            for n in ranked[:top_k]:
                score = lexical_score(query, self._candidate_text_with_body(n))
                distance = 1.0 - max(0.0, min(score, 1.0))
                hits.append(
                    RetrievedHit(
                        id=n["id"],
                        source=str(n.get("source", "")),
                        line_start=n.get("line_start"),
                        line_end=n.get("line_end"),
                        distance=distance,
                        title=str(n.get("title", "")),
                        breadcrumb=n.get("breadcrumb"),
                        score=score,
                        text=str(n.get("text", "")),
                    )
                )
            return hits

        frontier = [self.nodes["pageindex_root"]]
        visited: dict[str, dict] = {"pageindex_root": self.nodes["pageindex_root"]}

        for _ in range(max_depth):
            expanded: list[dict] = []
            for n in frontier:
                expanded.extend(self._children(n["id"]))
            if not expanded:
                break

            picked = self._pick_children(query, expanded, beam_width)
            frontier = picked
            for p in picked:
                visited[p["id"]] = p

        # Build final candidate pool from globally strong lexical matches plus visited nodes,
        # then let the LLM do the final selection/ranking.
        content_pool = [
            n
            for n in self.nodes.values()
            if n.get("source") and n.get("kind") in {"section", "top"} and (n.get("text") or n.get("summary"))
        ]
        content_pool = sorted(
            content_pool,
            key=lambda n: lexical_score(query, self._candidate_text_with_body(n)),
            reverse=True,
        )
        lexical_prefilter = content_pool[:80]

        merged_by_id: dict[str, dict] = {n["id"]: n for n in lexical_prefilter}
        for nid, node in visited.items():
            if nid == "pageindex_root":
                continue
            merged_by_id[nid] = node

        candidates = list(merged_by_id.values())
        ranked = sorted(
            candidates,
            key=lambda n: lexical_score(query, self._candidate_text_with_body(n)),
            reverse=True,
        )

        llm_ranked_ids = self._llm_rank_candidates(query, ranked[: self.final_candidate_limit], top_k)
        llm_ordered = False
        if llm_ranked_ids:
            id_to_node = {n["id"]: n for n in ranked}
            if self.llm_final_only:
                ranked = [id_to_node[nid] for nid in llm_ranked_ids if nid in id_to_node]
                llm_ordered = True
            else:
                llm_ranked = [id_to_node[nid] for nid in llm_ranked_ids if nid in id_to_node]
                # Keep LLM choices, but guard against obvious off-topic picks.
                kept = [
                    n
                    for n in llm_ranked
                    if lexical_score(query, self._candidate_text_with_body(n)) >= 0.35
                ]
                picked_ids = {n["id"] for n in kept}
                if len(kept) < top_k:
                    for n in ranked:
                        if n["id"] in picked_ids:
                            continue
                        kept.append(n)
                        picked_ids.add(n["id"])
                        if len(kept) >= top_k:
                            break
                ranked = kept
                llm_ordered = True

        hits: list[RetrievedHit] = []
        ranked_top = ranked[:top_k]
        denom = max(1, len(ranked_top) - 1)
        for idx, n in enumerate(ranked_top):
            if llm_ordered and self.use_llm:
                # In LLM mode, final ordering comes from the model, so use a
                # rank-based score to reflect the selected order.
                score = 1.0 if len(ranked_top) == 1 else 1.0 - (idx / denom)
            else:
                score = lexical_score(query, self._candidate_text(n))
            distance = 1.0 - max(0.0, min(score, 1.0))
            hits.append(
                RetrievedHit(
                    id=n["id"],
                    source=str(n.get("source", "")),
                    line_start=n.get("line_start"),
                    line_end=n.get("line_end"),
                    distance=distance,
                    title=str(n.get("title", "")),
                    breadcrumb=n.get("breadcrumb"),
                    score=score,
                    text=str(n.get("text", "")),
                )
            )

        return hits
