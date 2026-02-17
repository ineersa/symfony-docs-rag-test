# Reranker Fine-tuning & Local Optimization Plan

**Created:** February 2026
**Goal:** Optimize retrieval for local developer machines (MacBook/Laptop/No-GPU) while maintaining high accuracy for Symfony docs.

## 1. The Strategy: "Fat Build, Thin Client"

Move heavy context processing (Summarization) to a centralized build step, and keep local inference lightweight (Vector Search + ONNX Reranking).

### A. Centralized Build (The "Distro" Model)

- **Action**: Pre-compute `data/pageindex/nodes.jsonl` using high-end GPUs (e.g., Qwen-4B, batch size 16k).
- **Artifact**: A ~20MB compressed JSONL file containing rich summaries and tree structure.
- **Delivery**: Client downloads this artifact on startup if missing/outdated.

### B. Local Client (The "Player" Model)

- **Vector Search**: ChromaDB / SQLite-VSS (Low RAM).
- **Retrieval**: `retrieve.py` or new `local_retrieve.py`.
- **Reranking**: **ONNX Runtime (Int8 Quantized)**.
- **Generation**: **LiquidAI LFM2.5-1.2B-Instruct (ONNX)**.
  - _Backup_: Qwen2.5-1.5B-Instruct (if Liquid ONNX fails on specific hardware).

## 2. Reranker Architecture

### Target Model

- **Model**: `jinaai/jina-reranker-v2-base-multilingual`
- **Why**:
  - **Long Context**: Supports 8192 tokens (critical for documentation chunks).
  - **Size**: ~550MB (FP32) -> **~140MB (Int8 ONNX)**.
  - **Performance**: Comparable to BGE-Large for code tasks.
  - **Tech Stack**: ALIBI position embeddings supported in ONNX Runtime.

### Fine-Tuning Data Strategy

- **Dataset Source**:
  - **Positive**: `data/benchmark/questions.jsonl` (~900 items).
  - **Validation**: `data/benchmark/questions.gold.jsonl` (88 high-quality items).
- **Hard Negative Mining**:
  - Run current retrieval stack (BGE-Large or current baseline) against the training set.
  - Collect top-k incorrectly retrieved chunks as "Hard Negatives".
  - Mix: 1 Positive + 4 Hard Negatives + 4 Random Negatives.

### Implementation Next Steps

1.  **Generate Triplets**: Create `data/reranker_train.jsonl` using `benchmark_run.py` mining.
2.  **Fine-Tune**: Use Unsloth or SentenceTransformers to fine-tune `jina-v2-base`.
3.  **Export**: Convert to ONNX with dynamic quantization.
4.  **Integrate**: Replace `BGEReranker` with `OnnxReranker` class in `rerank_common.py`.

## 3. Tree Traversal Optimization

- **Problem**: Full tree traversal requires many LLM calls.
- **Solution**: "Hybrid-Lite"
  - Use Vector Search (with new Reranker) to find top 3 "Leaf Nodes".
  - Traverse _UP_ to parents for context, instead of _DOWN_ from root.
  - Use **LiquidAI LFM2.5-1.2B-Instruct** for final answer generation (Linear complexity context processing = fast summaries).
