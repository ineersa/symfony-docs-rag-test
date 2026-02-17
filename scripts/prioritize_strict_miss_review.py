#!/usr/bin/env python3
"""Prioritize suspicious strict-miss question/answer pairs for manual review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WEIGHTS = {
    "quote_span_mismatch": 5,
    "missing_required_entity": 4,
    "missing_backticked_terms": 3,
    "low_token_overlap": 1,
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def rank_row(row: dict) -> tuple[int, int]:
    reasons = row.get("suspicious_reasons", []) or []
    score = sum(WEIGHTS.get(reason, 0) for reason in reasons)
    overlap = float(row.get("overlap_ratio", 0.0) or 0.0)
    if overlap == 0.0:
        score += 1
    return score, len(reasons)


def suggested_action(reasons: list[str]) -> str:
    if "quote_span_mismatch" in reasons:
        return "fix line span and/or answer quote"
    if "missing_required_entity" in reasons:
        return "rewrite question or adjust required entity"
    if "missing_backticked_terms" in reasons:
        return "check backticked term relevance"
    return "manual semantic check"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritize suspicious strict-miss QA pairs")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/benchmark/hard_sets/strict_miss_top5_qa_quality.jsonl"),
        help="Input QA quality report",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Number of highest-priority rows to output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/hard_sets/strict_miss_top5_review_priority.jsonl"),
        help="Output prioritized JSONL",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.report)
    suspicious = [row for row in rows if row.get("status") == "suspicious"]

    ranked: list[dict] = []
    for row in suspicious:
        score, reason_count = rank_row(row)
        out = {
            "id": row.get("id"),
            "source_file": row.get("source_file"),
            "question": row.get("question"),
            "answer_excerpt": (str(row.get("answer", ""))[:240]).replace("\n", " "),
            "answer_line_start": row.get("answer_line_start"),
            "answer_line_end": row.get("answer_line_end"),
            "overlap_ratio": row.get("overlap_ratio"),
            "quote_span_exact_match": row.get("quote_span_exact_match"),
            "suspicious_reasons": row.get("suspicious_reasons", []),
            "priority_score": score,
            "reason_count": reason_count,
            "suggested_action": suggested_action(row.get("suspicious_reasons", [])),
        }
        ranked.append(out)

    ranked.sort(
        key=lambda r: (
            -int(r.get("priority_score", 0)),
            -int(r.get("reason_count", 0)),
            float(r.get("overlap_ratio", 0.0) or 0.0),
            str(r.get("source_file", "")),
        )
    )

    top_rows = ranked[: max(0, args.top)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in top_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Suspicious total: {len(suspicious)}")
    print(f"Wrote top {len(top_rows)} rows: {args.output}")
    for row in top_rows[:15]:
        print(
            f"- {row['id']} | score={row['priority_score']} | {row['source_file']} | "
            f"reasons={','.join(row['suspicious_reasons'])}"
        )


if __name__ == "__main__":
    main()
