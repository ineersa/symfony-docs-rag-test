#!/usr/bin/env python3
"""Utilities to repair line ranges in PageIndex trees.

Docutils line attribution can under-report section end lines, especially around
inline markup and directive boundaries. This module fills sibling gaps so each
section ends right before the next sibling section starts (same source file).
"""

from __future__ import annotations

from typing import Any


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def repair_tree_line_ranges(node: dict[str, Any], *, source_line_counts: dict[str, int] | None = None) -> None:
    """Repair in-place line_end values for section-like sibling nodes.

    For adjacent siblings from the same source with known starts, if the left
    node's line_end is missing or before ``next.line_start - 1``, extend it to
    that value.
    """

    children = node.get("children", []) or []
    for child in children:
        if isinstance(child, dict):
            repair_tree_line_ranges(child, source_line_counts=source_line_counts)

    for idx in range(len(children) - 1):
        left = children[idx]
        right = children[idx + 1]
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        if left.get("source") != right.get("source"):
            continue

        next_start = _as_int(right.get("line_start"))
        if next_start is None:
            continue

        target_end = next_start - 1
        if target_end < 1:
            continue

        current_end = _as_int(left.get("line_end"))
        if current_end is None or current_end < target_end:
            left["line_end"] = target_end

    # Extend the last same-source child to its container/file end when available.
    if not children:
        return
    last = children[-1]
    if not isinstance(last, dict):
        return

    parent_source = node.get("source")
    child_source = last.get("source")
    if not isinstance(parent_source, str) or child_source != parent_source:
        return

    target_end = _as_int(node.get("line_end"))
    if target_end is None and source_line_counts is not None:
        target_end = source_line_counts.get(parent_source)
    if target_end is None or target_end < 1:
        return

    current_end = _as_int(last.get("line_end"))
    if current_end is None or current_end < target_end:
        last["line_end"] = target_end
