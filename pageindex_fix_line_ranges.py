#!/usr/bin/env python3
"""Patch existing PageIndex artifacts to repair sibling line-range gaps.

Usage:
  uv run python pageindex_fix_line_ranges.py
  uv run python pageindex_fix_line_ranges.py --tree data/pageindex/tree.json --nodes data/pageindex/nodes.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pageindex_build import flatten_tree
from pageindex_linefix import repair_tree_line_ranges


def _collect_sources(node: dict) -> set[str]:
    out: set[str] = set()
    source = node.get("source")
    if isinstance(source, str) and source:
        out.add(source)
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            out.update(_collect_sources(child))
    return out


def _build_source_line_counts(docs_dir: Path, sources: set[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src in sources:
        p = docs_dir / src
        if not p.is_file():
            continue
        raw = p.read_text(encoding="utf-8", errors="replace")
        counts[src] = len(raw.splitlines())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair line ranges in existing pageindex artifacts")
    parser.add_argument("--tree", type=Path, default=Path("data/pageindex/tree.json"), help="Path to tree.json")
    parser.add_argument("--nodes", type=Path, default=Path("data/pageindex/nodes.jsonl"), help="Path to nodes.jsonl")
    parser.add_argument("--docs-dir", type=Path, default=Path("symfony-docs"), help="Docs root for source line counts")
    args = parser.parse_args()

    with open(args.tree, encoding="utf-8") as f:
        payload = json.load(f)

    tree = payload.get("tree")
    if not isinstance(tree, dict):
        raise SystemExit(f"Invalid tree structure in {args.tree}")

    source_line_counts = _build_source_line_counts(args.docs_dir, _collect_sources(tree))
    repair_tree_line_ranges(tree, source_line_counts=source_line_counts)

    flat_nodes: list[dict] = []
    flatten_tree(tree, parent_id=None, depth=0, out=flat_nodes)

    with open(args.tree, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    args.nodes.parent.mkdir(parents=True, exist_ok=True)
    with open(args.nodes, "w", encoding="utf-8") as f:
        for row in flat_nodes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Updated tree: {args.tree}")
    print(f"Updated nodes: {args.nodes}")
    print(f"Nodes: {len(flat_nodes)}")


if __name__ == "__main__":
    main()
