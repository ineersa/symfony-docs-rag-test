#!/usr/bin/env python3
"""Build a QA-link quality report for IDs from strict_miss_top5.jsonl.

For each strict-miss ID, this script:
- loads the matched question record from questions.jsonl;
- resolves includes for the source file;
- extracts the answer text using answer_line_start/answer_line_end;
- scores whether question and extracted answer are likely connected.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

INCLUDE_RE = re.compile(r"^\.\.\s+include::\s+(.+)$", re.MULTILINE)


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "using",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
    "symfony",
    "documentation",
    "docs",
    "guide",
}


def resolve_includes(text: str, file_path: Path, docs_root: Path, resolved: set[str]) -> str:
    """Recursively replace ``.. include::`` directives with file contents."""

    def _replace(match: re.Match) -> str:
        ref = match.group(1).strip()
        if ref.startswith("/"):
            inc_path = docs_root / ref.lstrip("/")
        else:
            inc_path = file_path.parent / ref
        inc_path = inc_path.resolve()

        if not inc_path.is_file():
            return match.group(0)

        resolved.add(str(inc_path))
        inc_text = inc_path.read_text(encoding="utf-8", errors="replace")
        return resolve_includes(inc_text, inc_path, docs_root, resolved)

    return INCLUDE_RE.sub(_replace, text)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_./:-]+", text)]
    out: list[str] = []
    for token in tokens:
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        out.append(token)
    return out


def parse_jsonl_indexed(path: Path) -> tuple[dict[str, dict], int]:
    data: dict[str, dict] = {}
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            rec_id = obj.get("id")
            if isinstance(rec_id, str):
                data[rec_id] = obj
    return data, malformed


def strict_ids(path: Path, from_line: int, to_line: int) -> list[str]:
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if i < from_line:
                continue
            if to_line > 0 and i > to_line:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec_id = obj.get("id")
            if isinstance(rec_id, str):
                ids.append(rec_id)
    return ids


def backticked_terms(question: str) -> list[str]:
    return [m.strip() for m in re.findall(r"`([^`]+)`", question) if m.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Report question-answer connection quality for strict_miss IDs")
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
        help="Path to questions.jsonl",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("symfony-docs"),
        help="Path to docs root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/hard_sets/strict_miss_top5_qa_quality.jsonl"),
        help="Output report JSONL",
    )
    parser.add_argument("--from-line", type=int, default=1, help="Start line in strict-miss file (1-based)")
    parser.add_argument("--to-line", type=int, default=0, help="End line in strict-miss file inclusive, 0=EOF")
    parser.add_argument(
        "--suspicious-threshold",
        type=float,
        default=0.08,
        help="Minimum question-term overlap ratio; below becomes suspicious",
    )
    args = parser.parse_args()

    docs_root = args.docs_dir.resolve()
    q_by_id, malformed = parse_jsonl_indexed(args.questions)
    ids = strict_ids(args.strict_miss, args.from_line, args.to_line)

    rows: list[dict] = []
    for rec_id in ids:
        row: dict = {"id": rec_id}
        q = q_by_id.get(rec_id)
        if not q:
            row["status"] = "missing_question"
            rows.append(row)
            continue

        question = str(q.get("question", ""))
        source_file = str(q.get("source_file", ""))
        start = int(q.get("answer_line_start", 0) or 0)
        end = int(q.get("answer_line_end", 0) or 0)
        quote = str(q.get("answer_quote", "") or "")
        required_entity = q.get("required_entity")

        row.update(
            {
                "question": question,
                "source_file": source_file,
                "answer_line_start": start,
                "answer_line_end": end,
            }
        )

        src = (docs_root / source_file).resolve()
        if not src.is_file():
            row["status"] = "source_missing"
            rows.append(row)
            continue

        raw = src.read_text(encoding="utf-8", errors="replace")
        resolved = resolve_includes(raw, src, docs_root, set())
        lines = resolved.splitlines()

        if not (1 <= start <= end <= len(lines)):
            row["status"] = "span_out_of_bounds"
            row["resolved_line_count"] = len(lines)
            rows.append(row)
            continue

        answer = "\n".join(lines[start - 1 : end])
        row["answer"] = answer
        row["quote_span_exact_match"] = normalize_ws(answer) == normalize_ws(quote)

        q_terms = tokenize(question)
        a_terms = set(tokenize(answer))
        overlap_terms = sorted(set(q_terms).intersection(a_terms))
        overlap_ratio = (len(overlap_terms) / len(set(q_terms))) if q_terms else 0.0
        row["q_term_count"] = len(set(q_terms))
        row["overlap_terms"] = overlap_terms
        row["overlap_ratio"] = round(overlap_ratio, 4)

        ticks = backticked_terms(question)
        ticks_present = [t for t in ticks if t.lower() in answer.lower()]
        row["backticked_terms"] = ticks
        row["backticked_terms_present"] = ticks_present

        entity_ok = True
        entity_check_enabled = False
        if isinstance(required_entity, str) and required_entity.strip():
            kind = str(q.get("kind", "")).strip().lower()
            # Treat required_entity as strict only for explicit entity questions
            # where the entity is actually part of the question text.
            entity_check_enabled = kind == "entity" and required_entity.lower() in question.lower()
            row["required_entity"] = required_entity
            row["required_entity_check_enabled"] = entity_check_enabled
            if entity_check_enabled:
                entity_ok = required_entity.lower() in answer.lower()
                row["required_entity_present"] = entity_ok

        suspicious_reasons: list[str] = []
        if not row["quote_span_exact_match"]:
            suspicious_reasons.append("quote_span_mismatch")
        if ticks and not ticks_present:
            suspicious_reasons.append("missing_backticked_terms")
        if entity_check_enabled and not entity_ok:
            suspicious_reasons.append("missing_required_entity")
        if len(set(q_terms)) >= 4 and overlap_ratio < args.suspicious_threshold:
            suspicious_reasons.append("low_token_overlap")

        row["suspicious_reasons"] = suspicious_reasons
        row["status"] = "suspicious" if suspicious_reasons else "ok"
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    total = len(rows)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    suspicious = sum(1 for r in rows if r.get("status") == "suspicious")
    missing_q = sum(1 for r in rows if r.get("status") == "missing_question")
    missing_src = sum(1 for r in rows if r.get("status") == "source_missing")
    oob = sum(1 for r in rows if r.get("status") == "span_out_of_bounds")

    print(f"Checked: {total}")
    print(f"OK: {ok}")
    print(f"Suspicious: {suspicious}")
    print(f"Missing question: {missing_q}")
    print(f"Missing source: {missing_src}")
    print(f"Out-of-bounds span: {oob}")
    if malformed:
        print(f"Warning: skipped {malformed} malformed questions.jsonl lines while indexing")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
