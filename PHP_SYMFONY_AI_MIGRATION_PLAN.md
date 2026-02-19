# PHP Symfony AI Migration Plan (V1)

Date: 2026-02-18

## Objective

Migrate this project from Python scripts to a PHP/Symfony stack, primarily to explore `symfony/ai`, while preserving practical RAG quality and keeping local-first operation.

This document captures the agreed V1 architecture and scope. It is planning-only (no implementation details are executed here).

## Locked V1 Scope

1. Runtime stack:
   - Symfony + FrankenPHP
   - Symfony AI components/bundle for model/store integration
   - SQLite for app state and job metadata
   - ChromaDB server for vector storage (persisted on local disk)
   - llama.cpp services in Docker Compose

2. Retrieval behavior:
   - BM25 + vector retrieval
   - RRF fusion
   - Optional reranker step (custom integration)
   - No tree expansion in V1
   - No lexical heuristic branch in V1

3. Ingestion behavior:
   - Source = git repository URL or local directory
   - Build structured doc tree + section anchors
   - Generate summaries
   - Embed summaries into Chroma
   - Build BM25 corpus/index

4. Citation format:
   - `file + section/anchor` (line-level citations are explicitly out of scope for V1)

5. UX:
   - Web UI + CLI coexist
   - Web is primary for end-user usability
   - CLI remains for power users/automation
   - Job progress via polling or SSE (no websocket requirement)

6. Packaging/distribution:
   - Distribute prebuilt tree + summaries only
   - Do not distribute embeddings/Chroma snapshot in V1

7. MCP:
   - Deferred to later phase

## Parser Strategy

Primary parser: `doctrine/rst-parser`.

Compatibility approach:
- Use Symfony Docs Builder (`symfony-tools/docs-builder`) as implementation reference for Symfony-specific directives/reference behavior.
- Build an app-owned compatibility adapter layer.
- Avoid runtime lock-in to docs-builder internals (it is documented as an internal tool with no BC guarantees).

## High-Level Architecture

```text
+-------------------+        +------------------------------+
| Browser Web UI    | -----> | Symfony App (FrankenPHP)    |
+-------------------+        | - API controllers            |
                             | - Console commands           |
+-------------------+        | - Ingestion orchestrator     |
| CLI               | -----> | - Retrieval service          |
+-------------------+        +--------------+---------------+
                                            |
                 +--------------------------+--------------------------+
                 |                          |                          |
                 v                          v                          v
          +-------------+            +-------------+          +------------------+
          | SQLite      |            | ChromaDB    |          | llama.cpp APIs   |
          | libraries,  |            | summaries   |          | embed/summary/   |
          | snapshots,  |            | embeddings  |          | rerank/generate  |
          | jobs, logs  |            |             |          | (separate svc)   |
          +-------------+            +-------------+          +------------------+
```

## Docker Compose Topology

Planned services:
- `app` (FrankenPHP + Symfony HTTP)
- `worker` (Symfony Messenger consumer)
- `chroma` (Chroma server with mounted local volume)
- `llama-embed`
- `llama-summary`
- `llama-rerank`
- `llama-generate` (optional)

All model switching should be environment-variable driven so users can change models without code edits.

## Data and Artifact Model

### SQLite tables (minimum)

- `libraries`
  - identity, source type/url/path, status
- `snapshots`
  - library linkage, version, active flag, manifest metadata
- `jobs`
  - type, status, progress, start/end timestamps
- `job_events`
  - structured logs for UI progress/history
- `settings`
  - runtime knobs/defaults

### Snapshot artifacts

- `tree.json` (or equivalent hierarchy artifact)
- `nodes.jsonl` (section-level flattened records)
- `summaries.jsonl`
- `manifest.json`

`manifest.json` should include parser/model versions and compatibility metadata.

## Ingestion Flow

```text
Create library (git/local)
 -> create ingestion job
 -> fetch source files
 -> parse RST + directives compatibility layer
 -> build section tree + stable anchors
 -> generate summaries
 -> build BM25 corpus/index
 -> embed summaries and upsert to Chroma
 -> persist artifacts + manifest
 -> mark snapshot ACTIVE
```

## Retrieval Flow

```text
query
 -> vector branch (embed query -> Chroma topN)
 -> BM25 branch (text search topN)
 -> RRF fusion
 -> optional rerank over fused shortlist
 -> topK result formatting
 -> return snippet previews + file/section anchors
 -> optional answer generation
```

## Citation and Result Contract (V1)

Each hit should contain at least:
- `source_file`
- `section_title`
- `section_anchor`
- `preview_text`
- retrieval score metadata (vector/bm25/fused/rerank as available)

No line spans required in V1.

## Reliability and Fallbacks

- If reranker is unavailable, continue with fused ranking.
- If generation model is unavailable, return retrieval-only output.
- Unknown directives should degrade gracefully (warn/log, do not crash ingestion).
- Long-running ingestion must be resumable/retriable per stage.

## UI and CLI Responsibilities

Web UI:
- library setup/config
- ingestion run/monitoring
- retrieval playground
- tuning knobs with sensible defaults

CLI:
- ingestion and maintenance commands
- diagnostics and smoke checks
- automation-friendly workflows

Both paths must call the same internal services to avoid behavioral drift.

## Risks and Mitigations

1. RST compatibility gaps
   - Mitigation: adapter layer based on docs-builder behavior + fixtures from Symfony docs corpus.

2. Anchor collisions/instability
   - Mitigation: deterministic slug rules + duplicate suffixing strategy.

3. Model endpoint mismatches (especially rerank)
   - Mitigation: capability checks at startup + feature flags + graceful fallback.

4. Resource pressure running multiple llama.cpp services
   - Mitigation: allow selective service enablement and sensible default models.

5. Parser/runtime dependency volatility
   - Mitigation: pin versions and isolate parser integration behind interfaces.

## Phased Rollout

Phase 0 - Foundation
- Symfony skeleton, Docker Compose, config model.

Phase 1 - Ingestion Core
- parser + tree/anchor extraction + summaries + artifacts.

Phase 2 - Retrieval Core
- BM25 + vector + RRF + optional rerank.

Phase 3 - UX
- Web UI for libraries/jobs/query + CLI parity commands.

Phase 4 - Import/Export
- tree/summaries snapshot import/export and compatibility checks.

Phase 5 - Hardening
- smoke benchmarks, reliability tests, operability improvements.

## Explicit Non-Goals (V1)

- Tree expansion in retrieval
- Line-aware citations
- MCP tool server rollout
- Distribution of prebuilt vector embeddings/Chroma snapshots

## Future Extensions (Post-V1)

- MCP server (`search_docs`, `get_snippet`, etc.)
- Tree expansion branch with optional hybrid traversal
- Packaged embedding snapshots if portability/versioning is solved
- Strict line-citation support
- Upstream PR(s) for reranker integration ergonomics
