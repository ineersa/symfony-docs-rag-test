#!/usr/bin/env python3
"""Validate strict_miss_top5 question spans against include-resolved source text.

This script cross-checks IDs from a strict miss file with benchmark questions and
verifies whether each question's answer span still matches the answer quote when
the source file is resolved with ``.. include::`` expansion.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


INCLUDE_RE = re.compile(r"^\.\.\s+include::\s+(.+)$")


@dataclass
class ResolvedLine:
    text: str
    origin_file: str
    origin_line: int


def _normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_with_pos_map(value: str) -> tuple[str, list[int]]:
    """Normalize whitespace and keep mapping from normalized char -> source index."""
    chars: list[str] = []
    pos_map: list[int] = []
    previous_space = True

    for idx, ch in enumerate(value):
        if ch.isspace():
            if not previous_space and chars:
                chars.append(" ")
                pos_map.append(idx)
            previous_space = True
            continue

        chars.append(ch)
        pos_map.append(idx)
        previous_space = False

    if chars and chars[-1] == " ":
        chars.pop()
        pos_map.pop()

    return "".join(chars), pos_map


def _line_number_from_pos(text: str, pos: int) -> int:
    if pos <= 0:
        return 1
    return text.count("\n", 0, pos) + 1


def find_line_span(text: str, quote: str) -> tuple[int, int] | None:
    quote = quote.strip()
    if not quote:
        return None

    idx = text.find(quote)
    if idx != -1:
        start = _line_number_from_pos(text, idx)
        end = _line_number_from_pos(text, idx + len(quote))
        return start, end

    normalized_text, pos_map = _normalize_with_pos_map(text)
    normalized_quote = _normalize_ws(quote)
    if not normalized_text or not normalized_quote:
        return None

    n_idx = normalized_text.find(normalized_quote)
    if n_idx == -1:
        return None

    start_pos = pos_map[n_idx]
    end_norm_idx = n_idx + len(normalized_quote) - 1
    end_pos = pos_map[end_norm_idx] + 1

    start = _line_number_from_pos(text, start_pos)
    end = _line_number_from_pos(text, min(len(text), end_pos))
    return start, end


def find_exact_spans(text: str, quote: str) -> list[tuple[int, int]]:
    quote = quote.strip()
    if not quote:
        return []

    spans: list[tuple[int, int]] = []
    start_at = 0
    while True:
        idx = text.find(quote, start_at)
        if idx == -1:
            break
        start = _line_number_from_pos(text, idx)
        end = _line_number_from_pos(text, idx + len(quote))
        spans.append((start, end))
        start_at = idx + 1
    return spans


def resolve_includes_with_map(file_path: Path, docs_root: Path, stack: set[Path]) -> list[ResolvedLine]:
    if file_path in stack:
        return []

    stack.add(file_path)
    out: list[ResolvedLine] = []
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    for idx, line in enumerate(lines, start=1):
        match = INCLUDE_RE.match(line)
        if not match:
            out.append(ResolvedLine(text=line, origin_file=str(file_path), origin_line=idx))
            continue

        ref = match.group(1).strip()
        if ref.startswith("/"):
            inc_path = docs_root / ref.lstrip("/")
        else:
            inc_path = file_path.parent / ref
        inc_path = inc_path.resolve()

        if not inc_path.is_file() or inc_path in stack:
            out.append(ResolvedLine(text=line, origin_file=str(file_path), origin_line=idx))
            continue

        out.extend(resolve_includes_with_map(inc_path, docs_root, stack))

    stack.remove(file_path)
    return out


def read_jsonl_by_id(path: Path) -> tuple[dict[str, dict], int]:
    out: dict[str, dict] = {}
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            rec_id = record.get("id")
            if isinstance(rec_id, str):
                out[rec_id] = record
    return out, malformed


def main() -> None:
    parser = argparse.ArgumentParser(description="Check strict_miss_top5 alignment against include-resolved files")
    parser.add_argument(
        "--strict-miss",
        type=Path,
        default=Path("data/benchmark/hard_sets/strict_miss_top5.jsonl"),
        help="Path to strict_miss_top5.jsonl",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/benchmark/questions.jsonl"),
        help="Path to benchmark questions JSONL",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("symfony-docs"),
        help="Symfony docs root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/hard_sets/strict_miss_top5_line_check.jsonl"),
        help="Validation report JSONL output path",
    )
    parser.add_argument(
        "--from-line",
        type=int,
        default=1,
        help="Start line (1-based) inside strict miss file",
    )
    parser.add_argument(
        "--to-line",
        type=int,
        default=0,
        help="End line (inclusive) inside strict miss file; 0 means EOF",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Update questions.jsonl line spans for mismatches when quote is found in resolved text",
    )
    parser.add_argument(
        "--write-report-only",
        action="store_true",
        help="When used with --fix, skip writing the analysis report JSONL",
    )
    parser.add_argument(
        "--fix-update-quote",
        action="store_true",
        help="When fixing, also rewrite answer_quote from resolved text at fixed span",
    )
    args = parser.parse_args()

    docs_root = args.docs_dir.resolve()
    questions_by_id, malformed = read_jsonl_by_id(args.questions)

    results: list[dict] = []
    auto_fixes: dict[str, tuple[int, int]] = {}
    quote_fixes: dict[str, str] = {}
    with args.strict_miss.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line_no < args.from_line:
                continue
            if args.to_line > 0 and line_no > args.to_line:
                break

            line = line.strip()
            if not line:
                continue
            miss = json.loads(line)
            rec_id = miss.get("id")
            if not isinstance(rec_id, str):
                continue

            row: dict = {
                "strict_line": line_no,
                "id": rec_id,
            }

            question = questions_by_id.get(rec_id)
            if not question:
                row["status"] = "missing_question"
                results.append(row)
                continue

            source_file = question.get("source_file")
            if not isinstance(source_file, str) or not source_file:
                row["status"] = "invalid_source_file"
                results.append(row)
                continue

            src_path = (docs_root / source_file).resolve()
            row["source_file"] = source_file
            if not src_path.is_file():
                row["status"] = "source_missing"
                row["resolved_source_path"] = str(src_path)
                results.append(row)
                continue

            resolved_lines = resolve_includes_with_map(src_path, docs_root, set())
            resolved_text = "\n".join(item.text for item in resolved_lines)
            total_lines = len(resolved_lines)

            start = int(question.get("answer_line_start", 0) or 0)
            end = int(question.get("answer_line_end", 0) or 0)
            quote = str(question.get("answer_quote", "") or "")

            row["answer_line_start"] = start
            row["answer_line_end"] = end
            row["resolved_total_lines"] = total_lines

            valid_span = 1 <= start <= end <= total_lines
            row["span_in_resolved_bounds"] = valid_span
            span_text = ""

            if valid_span:
                span_text = "\n".join(item.text for item in resolved_lines[start - 1 : end])
                row["span_exact_quote_match"] = span_text.strip() == quote.strip()
                row["span_normalized_quote_match"] = _normalize_ws(span_text) == _normalize_ws(quote)
                row["span_origin_start"] = {
                    "file": str(Path(resolved_lines[start - 1].origin_file).relative_to(docs_root)),
                    "line": resolved_lines[start - 1].origin_line,
                }
                row["span_origin_end"] = {
                    "file": str(Path(resolved_lines[end - 1].origin_file).relative_to(docs_root)),
                    "line": resolved_lines[end - 1].origin_line,
                }
            else:
                row["span_exact_quote_match"] = False
                row["span_normalized_quote_match"] = False

            exact_spans = find_exact_spans(resolved_text, quote)
            if exact_spans:
                found = min(exact_spans, key=lambda sp: abs(sp[0] - start))
            else:
                found = find_line_span(resolved_text, quote)
            row["quote_found_in_resolved"] = found is not None
            if found is not None:
                fstart, fend = found
                row["found_line_start"] = fstart
                row["found_line_end"] = fend
                found_text = "\n".join(item.text for item in resolved_lines[fstart - 1 : fend])
                row["found_origin_start"] = {
                    "file": str(Path(resolved_lines[fstart - 1].origin_file).relative_to(docs_root)),
                    "line": resolved_lines[fstart - 1].origin_line,
                }
                row["found_origin_end"] = {
                    "file": str(Path(resolved_lines[fend - 1].origin_file).relative_to(docs_root)),
                    "line": resolved_lines[fend - 1].origin_line,
                }
                quote_fixes[rec_id] = found_text

            row["status"] = (
                "ok"
                if row["span_in_resolved_bounds"] and row["span_normalized_quote_match"]
                else "mismatch"
            )

            if (
                row["status"] == "mismatch"
                and row.get("quote_found_in_resolved")
                and isinstance(row.get("found_line_start"), int)
                and isinstance(row.get("found_line_end"), int)
            ):
                auto_fixes[rec_id] = (int(row["found_line_start"]), int(row["found_line_end"]))
            elif row["status"] == "mismatch" and valid_span:
                quote_fixes[rec_id] = span_text

            results.append(row)

    if not args.write_report_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=True) + "\n")

    fixed = 0
    quote_rewritten = 0
    if args.fix and auto_fixes:
        updated_lines: list[str] = []
        with args.questions.open(encoding="utf-8") as handle:
            for line in handle:
                raw_line = line.rstrip("\n")
                stripped = raw_line.strip()
                if not stripped:
                    updated_lines.append(raw_line)
                    continue

                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    updated_lines.append(raw_line)
                    continue

                rec_id = record.get("id")
                touched = False
                if isinstance(rec_id, str) and rec_id in auto_fixes:
                    start, end = auto_fixes[rec_id]
                    if record.get("answer_line_start") != start or record.get("answer_line_end") != end:
                        record["answer_line_start"] = start
                        record["answer_line_end"] = end
                        fixed += 1
                        touched = True

                if isinstance(rec_id, str) and args.fix_update_quote and rec_id in quote_fixes:
                    replacement = quote_fixes[rec_id]
                    if replacement and record.get("answer_quote") != replacement:
                        record["answer_quote"] = replacement
                        quote_rewritten += 1
                        touched = True

                if touched:
                    updated_lines.append(json.dumps(record, ensure_ascii=False))
                else:
                    updated_lines.append(raw_line)

        args.questions.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    ok = sum(1 for item in results if item.get("status") == "ok")
    mismatch = sum(1 for item in results if item.get("status") == "mismatch")
    missing_question = sum(1 for item in results if item.get("status") == "missing_question")
    source_missing = sum(1 for item in results if item.get("status") == "source_missing")
    fix_candidates = len(auto_fixes)

    print(f"Checked: {len(results)}")
    print(f"OK: {ok}")
    print(f"Mismatch: {mismatch}")
    print(f"Missing question by ID: {missing_question}")
    print(f"Missing source file: {source_missing}")
    print(f"Auto-fix candidates: {fix_candidates}")
    if args.fix:
        print(f"Applied fixes: {fixed}")
        if args.fix_update_quote:
            print(f"Rewritten answer_quote: {quote_rewritten}")
    if malformed:
        print(f"Warning: skipped {malformed} malformed question rows while indexing questions.jsonl")
    if not args.write_report_only:
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
