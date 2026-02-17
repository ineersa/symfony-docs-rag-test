#!/usr/bin/env python3
"""Re-anchor benchmark answer line spans using answer_quote against resolved docs text."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path


def _load_chunk_module(repo_root: Path):
    chunk_path = repo_root / "chunk.py"
    spec = importlib.util.spec_from_file_location("project_chunk", chunk_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {chunk_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line_from_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _normalize(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _find_span_by_quote(resolved_text: str, quote: str) -> tuple[int, int] | None:
    if not resolved_text or not quote:
        return None

    exact_idx = resolved_text.find(quote)
    if exact_idx >= 0:
        start = _line_from_offset(resolved_text, exact_idx)
        end = _line_from_offset(resolved_text, exact_idx + len(quote))
        return start, end

    src_lines = resolved_text.splitlines()
    src_nonempty: list[tuple[int, str]] = []
    for i, line in enumerate(src_lines, start=1):
        norm = _normalize(line)
        if norm:
            src_nonempty.append((i, norm))

    quote_nonempty = [_normalize(line) for line in quote.splitlines() if _normalize(line)]
    if not quote_nonempty or not src_nonempty:
        return None

    src_norm_only = [norm for _, norm in src_nonempty]
    n = len(quote_nonempty)
    for i in range(0, len(src_norm_only) - n + 1):
        if src_norm_only[i : i + n] == quote_nonempty:
            start = src_nonempty[i][0]
            end = src_nonempty[i + n - 1][0]
            return start, end

    first = quote_nonempty[0]
    last = quote_nonempty[-1]
    first_hits = [ln for ln, norm in src_nonempty if norm == first]
    last_hits = [ln for ln, norm in src_nonempty if norm == last]
    if not first_hits or not last_hits:
        return None

    best: tuple[int, int] | None = None
    best_span = 10**9
    for s in first_hits:
        for e in last_hits:
            if e < s:
                continue
            span = e - s
            if span < best_span:
                best_span = span
                best = (s, e)
    return best


def _resolved_source_text(source_file: str, docs_dir: Path, resolve_includes_fn) -> str:
    file_path = docs_dir / source_file
    raw_text = file_path.read_text(encoding="utf-8", errors="replace")
    return resolve_includes_fn(raw_text, file_path, docs_dir, set())


def reanchor_file(path: Path, docs_dir: Path, resolve_includes_fn) -> tuple[int, int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    cache: dict[str, str] = {}
    changed = 0
    unresolved = 0

    for row in rows:
        source_file = row.get("source_file")
        answer_quote = row.get("answer_quote")
        if not isinstance(source_file, str) or not isinstance(answer_quote, str):
            unresolved += 1
            continue

        if source_file not in cache:
            try:
                cache[source_file] = _resolved_source_text(source_file, docs_dir, resolve_includes_fn)
            except Exception:
                cache[source_file] = ""

        span = _find_span_by_quote(cache[source_file], answer_quote)
        if span is None:
            unresolved += 1
            continue

        start, end = span
        old_start = row.get("answer_line_start")
        old_end = row.get("answer_line_end")
        row["answer_line_start"] = start
        row["answer_line_end"] = end
        if old_start != start or old_end != end:
            changed += 1

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows), changed, unresolved


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    chunk_module = _load_chunk_module(repo_root)

    parser = argparse.ArgumentParser(description="Re-anchor question spans using answer_quote")
    parser.add_argument("files", nargs="+", type=Path, help="Question JSONL files to update in place")
    parser.add_argument("--docs-dir", type=Path, default=Path(chunk_module.DOCS_DIR), help="Symfony docs root")
    args = parser.parse_args()

    for file_path in args.files:
        total, changed, unresolved = reanchor_file(file_path, args.docs_dir, chunk_module.resolve_includes)
        print(f"{file_path}: total={total} changed={changed} unresolved={unresolved}")


if __name__ == "__main__":
    main()
