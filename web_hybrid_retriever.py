#!/usr/bin/env python3
"""Separate hybrid retriever for the web app (CPU transformers + ONNX reranker)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import chromadb
import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer

from bm25_common import BM25Index, reciprocal_rank_fusion
from pageindex_common import QUERY_PREFIX, RetrievedHit
from rerank_common import split_text_for_rerank


DEFAULT_CHROMA_DIR = Path("data/chroma")
DEFAULT_COLLECTION = "symfony_pageindex_summaries"
DEFAULT_QUERY_EMBED_MODEL = "nomic-ai/CodeRankEmbed"
DEFAULT_QUERY_EMBED_REVISION = "3c4b60807d71f79b43f3c4363786d9493691f8b1"
DEFAULT_RERANK_REPO = "onnx-community/bge-reranker-base-ONNX"
DEFAULT_RERANK_FILE = "onnx/model_int8.onnx"
DEFAULT_RERANK_BATCH_SIZE = 16


@dataclass
class _NodeEvidence:
    score: float = 0.0
    best_distance: float = 1.0
    hit_count: int = 0
    best_rank: int = 10_000


class TransformerCodeRankEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_QUERY_EMBED_MODEL,
        *,
        revision: str = DEFAULT_QUERY_EMBED_REVISION,
    ):
        local_model_dir = snapshot_download(
            repo_id=model_name,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.py",
                "*.safetensors",
                "*.bin",
            ],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            local_model_dir,
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_length = 8192

    @staticmethod
    def _enriched_prefixed_query(query: str) -> str:
        enriched = query
        if "symfony" not in query.lower():
            enriched = f"{query} (Symfony PHP framework)"
        return f"{QUERY_PREFIX}{enriched}"

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        if not queries:
            return []
        prefixed = [self._enriched_prefixed_query(q) for q in queries]
        with torch.inference_mode():
            encoded = self.tokenizer(
                prefixed,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out = self.model(**encoded)
            embeddings = out.last_hidden_state[:, 0, :]
            vectors = embeddings.detach().cpu().float().numpy()
        return vectors.tolist()


class ONNXBGEReranker:
    def __init__(
        self,
        *,
        onnx_path: str | None = None,
        model_repo: str = DEFAULT_RERANK_REPO,
        model_file: str = DEFAULT_RERANK_FILE,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.batch_size = DEFAULT_RERANK_BATCH_SIZE
        model_path = onnx_path
        if not model_path:
            model_path = hf_hub_download(repo_id=model_repo, filename=model_file)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = {inp.name for inp in self.session.get_inputs()}
        self.output_name = self.session.get_outputs()[0].name

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        out_scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            pairs = [(query, p) for p in batch]
            enc = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            feeds: dict[str, np.ndarray] = {}
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in self.input_names and key in enc:
                    feeds[key] = enc[key].astype(np.int64)

            outputs = self.session.run([self.output_name], feeds)
            logits = np.asarray(outputs[0]).reshape(-1)
            scores = 1.0 / (1.0 + np.exp(-logits))
            out_scores.extend(float(x) for x in scores)
        return out_scores


class WebHybridRetriever:
    def __init__(
        self,
        nodes: dict[str, dict],
        *,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection: str = DEFAULT_COLLECTION,
        query_embed_model: str = DEFAULT_QUERY_EMBED_MODEL,
        query_embed_revision: str = DEFAULT_QUERY_EMBED_REVISION,
        use_reranker: bool = True,
        reranker_onnx_path: str | None = None,
        reranker_repo: str = DEFAULT_RERANK_REPO,
        reranker_file: str = DEFAULT_RERANK_FILE,
        vector_top_n: int = 20,
        candidate_cap: int = 15,
        neighbor_depth: int = 1,
        siblings_per_node: int = 2,
        candidate_file_roots: int = 5,
        bm25_top_n: int = 20,
        rrf_k: int = 60,
        rrf_vector_weight: float = 1.0,
        rrf_bm25_weight: float = 1.0,
        rerank_chunk_chars: int = 1500,
        rerank_chunk_overlap: int = 500,
        logger: Callable[[str], None] | None = None,
    ):
        self.nodes = nodes
        self._logger = logger
        self._log("initializing query embedder")
        self.embedder = TransformerCodeRankEmbedder(query_embed_model, revision=query_embed_revision)
        self.use_reranker = bool(use_reranker)
        self.vector_top_n = max(1, vector_top_n)
        self.candidate_cap = max(1, candidate_cap)
        self.neighbor_depth = max(0, neighbor_depth)
        self.siblings_per_node = max(0, siblings_per_node)
        self.candidate_file_roots = max(0, candidate_file_roots)
        self.bm25_top_n = max(1, bm25_top_n)
        self.rrf_k = max(1, rrf_k)
        self.rrf_vector_weight = max(0.0, rrf_vector_weight)
        self.rrf_bm25_weight = max(0.0, rrf_bm25_weight)
        self.rerank_chunk_chars = max(64, int(rerank_chunk_chars))
        self.rerank_chunk_overlap = max(0, min(int(rerank_chunk_overlap), self.rerank_chunk_chars - 1))

        self.reranker = None
        if self.use_reranker:
            self._log("initializing ONNX reranker")
            self.reranker = ONNXBGEReranker(
                onnx_path=reranker_onnx_path,
                model_repo=reranker_repo,
                model_file=reranker_file,
            )

        self._log("opening Chroma collection")
        chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = chroma_client.get_collection(collection)

        self.file_node_by_source = self._index_file_roots()
        self._log("building BM25 index")
        self.bm25_node_ids, self.bm25 = self._build_bm25_index()
        self._log("retriever ready")

    def _log(self, message: str) -> None:
        if self._logger:
            self._logger(f"[retriever] {message}")

    def _index_file_roots(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for n in self.nodes.values():
            if n.get("kind") == "file" and n.get("source"):
                out[str(n["source"])] = str(n["id"])
        return out

    def _node_bm25_text(self, node: dict[str, Any]) -> str:
        return str(node.get("text") or "")

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
        self._log("embedding query")
        embeddings = self.embedder.embed_queries([query])
        if not embeddings:
            return {}, {}

        evidence: dict[str, _NodeEvidence] = {}
        file_scores: dict[str, float] = {}

        self._log("searching vector index")
        result = self.collection.query(
            query_embeddings=[embeddings[0]],
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
            per_hit_score = (0.7 * closeness) + (0.3 * rank_weight)

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
        for _ in range(self.neighbor_depth):
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

    def _candidate_passages(self, candidate: dict) -> list[str]:
        text = str(candidate.get("text") or "")
        chunks = split_text_for_rerank(
            text,
            chunk_chars=self.rerank_chunk_chars,
            overlap_chars=self.rerank_chunk_overlap,
        )
        if not chunks:
            return [text] if text.strip() else []
        return [chunk for chunk in chunks if chunk.strip()]

    def _rerank_candidates(self, query: str, candidates: list[dict]) -> dict[str, float]:
        if not self.reranker or not candidates:
            return {}

        passages: list[str] = []
        owner_ids: list[str] = []
        for candidate in candidates:
            owner_id = str(candidate.get("id", ""))
            for passage in self._candidate_passages(candidate):
                passages.append(passage)
                owner_ids.append(owner_id)
        if not passages:
            return {}

        self._log(f"reranking {len(passages)} passages (batch={DEFAULT_RERANK_BATCH_SIZE})")
        scores = self.reranker.score(query, passages)
        by_id: dict[str, float] = {}
        for owner_id, score in zip(owner_ids, scores):
            prev = by_id.get(owner_id)
            if prev is None or float(score) > prev:
                by_id[owner_id] = float(score)
        return by_id

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedHit]:
        self._log(f"retrieve start (top_k={top_k})")
        evidence, file_scores = self._vector_shortlist(query)

        seed_ranked = sorted(
            evidence.items(),
            key=lambda x: (x[1].score, x[1].hit_count, -x[1].best_distance),
            reverse=True,
        )
        seed_scores = {node_id: ev.score for node_id, ev in seed_ranked[: self.candidate_cap]}
        expanded_scores = self._expand_neighbors(seed_scores, file_scores)

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

        self._log("running BM25 and RRF fusion")
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

        rerank_scores = self._rerank_candidates(query, candidates) if self.use_reranker else {}
        if rerank_scores:
            candidate_pos = {str(n.get("id", "")): idx for idx, n in enumerate(candidates)}
            reranked_ids = sorted(
                [str(n.get("id", "")) for n in candidates],
                key=lambda nid: (rerank_scores.get(nid, -1.0), -candidate_pos.get(nid, 0)),
                reverse=True,
            )
            by_id = {n["id"]: n for n in candidates}
            final_nodes = [by_id[nid] for nid in reranked_ids if nid in by_id]
        else:
            final_nodes = candidates

        hits: list[RetrievedHit] = []
        max_fused_score = max((final_scores.get(str(n.get("id")), 0.0) for n in final_nodes), default=0.0)
        for node in final_nodes[:top_k]:
            node_id = str(node.get("id", ""))
            rerank_score = rerank_scores.get(node_id)
            if rerank_score is not None:
                score = rerank_score
            else:
                fused_score = final_scores.get(node_id, 0.0)
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
        self._log(f"retrieve done ({len(hits)} hits)")
        return hits
