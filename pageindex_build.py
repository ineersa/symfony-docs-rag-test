#!/usr/bin/env python3
"""Build a PageIndex-style hierarchical index from Symfony RST docs.

Outputs:
- data/pageindex/tree.json
- data/pageindex/nodes.jsonl
"""

import argparse
import asyncio
import hashlib
import json
import re
from pathlib import Path

from docutils import nodes
from docutils.core import publish_doctree
from openai import AsyncOpenAI
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from chunk import DOCS_DIR, collect_rst_files, extract_section_content, node_line_range, resolve_includes

OUTPUT_DIR = Path("data/pageindex")
TREE_FILE = OUTPUT_DIR / "tree.json"
NODES_FILE = OUTPUT_DIR / "nodes.jsonl"
DEFAULT_SUMMARY_BASE_URL = "http://localhost:8052/v1"
DEFAULT_SUMMARY_MODEL = "local-model"

TOCTREE_RE = re.compile(r"^\.\.\s+toctree::\s*$")


def _node_id(kind: str, source: str, breadcrumb: str, line_start: int | None, line_end: int | None) -> str:
    raw = f"{kind}|{source}|{breadcrumb}|{line_start}|{line_end}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _summarize(text: str, max_chars: int = 320) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= max_chars:
        return one_line
    return one_line[: max_chars - 3].rstrip() + "..."


def _summary_prompt(text: str) -> str:
    return (
        "You are given part of a documentation page. "
        "Write a concise summary of the key points covered in this part.\n\n"
        f"Partial document text:\n{text}\n\n"
        "Return only the summary text."
    )


async def _generate_summary(
    client: AsyncOpenAI,
    *,
    model: str,
    text: str,
    input_chars: int,
    output_chars: int,
) -> str:
    clipped = (text or "")[:input_chars]
    if not clipped.strip():
        return ""
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _summary_prompt(clipped)}],
        temperature=0,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return ""
    return _summarize(content, max_chars=output_chars)


async def enrich_summaries_with_llm(
    rows: list[dict],
    *,
    base_url: str,
    model: str,
    concurrency: int,
    input_chars: int,
    output_chars: int,
    progress: Progress | None = None,
    task_id: TaskID | None = None,
) -> int:
    client = AsyncOpenAI(base_url=base_url, api_key="not-needed")
    semaphore = asyncio.Semaphore(max(1, concurrency))

    targets = [
        row for row in rows if row.get("kind") in {"section", "top"} and (row.get("text") or "").strip()
    ]

    async def worker(row: dict) -> None:
        try:
            async with semaphore:
                summary = await _generate_summary(
                    client,
                    model=model,
                    text=row.get("text") or "",
                    input_chars=input_chars,
                    output_chars=output_chars,
                )
            if summary:
                row["summary"] = summary
        except Exception:
            return

    tasks = [asyncio.create_task(worker(row)) for row in targets]
    for done in asyncio.as_completed(tasks):
        await done
        if progress is not None and task_id is not None:
            progress.advance(task_id)
    return len(targets)


def _apply_summary_map(node: dict, summary_by_id: dict[str, str]) -> None:
    node_id = node.get("id")
    if node_id in summary_by_id:
        node["summary"] = summary_by_id[node_id]
    for child in node.get("children", []):
        _apply_summary_map(child, summary_by_id)


def _progress() -> Progress:
    return Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=None),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def parse_toctree_links(file_path: Path, docs_dir: Path) -> list[str]:
    """Return normalized .rst relative paths referenced by toctree entries."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    links: list[str] = []
    i = 0

    while i < len(lines):
        if not TOCTREE_RE.match(lines[i]):
            i += 1
            continue

        i += 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if re.match(r"^\s{3,}:", line):
                i += 1
                continue
            if not re.match(r"^\s{3,}\S", line):
                break

            entry = line.strip()
            if "://" in entry:
                i += 1
                continue

            # normalize path without extension
            if entry.startswith("/"):
                stem = entry[1:]
            else:
                stem = str((file_path.parent / entry).relative_to(docs_dir)).replace("\\", "/")

            if "*" in stem:
                # Expand wildcard from docs root
                pattern = stem + ".rst"
                for matched in sorted(docs_dir.glob(pattern)):
                    if matched.is_file():
                        links.append(str(matched.relative_to(docs_dir)).replace("\\", "/"))
            else:
                if not stem.endswith(".rst"):
                    stem = stem + ".rst"
                links.append(stem)
            i += 1

    # preserve order, dedupe
    seen = set()
    ordered: list[str] = []
    for item in links:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def build_doc_structure(doctree: nodes.document, source_file: str) -> dict:
    """Build per-file section tree node with nested children."""
    root = {
        "id": _node_id("file", source_file, source_file, None, None),
        "kind": "file",
        "title": source_file,
        "source": source_file,
        "breadcrumb": source_file,
        "line_start": None,
        "line_end": None,
        "text": "",
        "summary": source_file,
        "children": [],
    }

    # top-level non-section content
    top_parts: list[str] = []
    top_nodes: list[nodes.Node] = []

    def process_section(section: nodes.section, parent_breadcrumb: list[str]) -> dict:
        title = ""
        for ch in section.children:
            if isinstance(ch, nodes.title):
                title = ch.astext()
                break
        breadcrumb_parts = parent_breadcrumb + [title] if title else parent_breadcrumb
        breadcrumb = " > ".join(breadcrumb_parts) if breadcrumb_parts else source_file

        body_text = _clean_text(extract_section_content(section))
        line_start, line_end = node_line_range(section)

        node = {
            "id": _node_id("section", source_file, breadcrumb, line_start, line_end),
            "kind": "section",
            "title": title or "(untitled section)",
            "source": source_file,
            "breadcrumb": breadcrumb,
            "line_start": line_start,
            "line_end": line_end,
            "text": body_text,
            "summary": _summarize(body_text) if body_text else (title or source_file),
            "children": [],
        }

        for child in section.children:
            if isinstance(child, nodes.section):
                node["children"].append(process_section(child, breadcrumb_parts))

        return node

    for child in doctree.children:
        if isinstance(child, nodes.section):
            root["children"].append(process_section(child, []))
        else:
            txt = child.astext().strip()
            if txt:
                top_parts.append(txt)
                top_nodes.append(child)

    if top_parts:
        top_text = _clean_text("\n\n".join(top_parts))
        line_start, line_end = None, None
        line_ranges = [node_line_range(n) for n in top_nodes]
        starts = [s for s, _ in line_ranges if isinstance(s, int)]
        ends = [e for _, e in line_ranges if isinstance(e, int)]
        if starts:
            line_start = min(starts)
        if ends:
            line_end = max(ends)

        top_node = {
            "id": _node_id("top", source_file, "(top-level)", line_start, line_end),
            "kind": "top",
            "title": "(top-level)",
            "source": source_file,
            "breadcrumb": "(top-level)",
            "line_start": line_start,
            "line_end": line_end,
            "text": top_text,
            "summary": _summarize(top_text),
            "children": [],
        }
        root["children"].insert(0, top_node)

    return root


def flatten_tree(node: dict, parent_id: str | None, depth: int, out: list[dict]) -> None:
    children = node.get("children", [])
    row = {
        "id": node["id"],
        "parent_id": parent_id,
        "children": [c["id"] for c in children],
        "depth": depth,
        "kind": node.get("kind"),
        "title": node.get("title"),
        "summary": node.get("summary", ""),
        "text": node.get("text", ""),
        "source": node.get("source"),
        "breadcrumb": node.get("breadcrumb"),
        "line_start": node.get("line_start"),
        "line_end": node.get("line_end"),
    }
    out.append(row)
    for child in children:
        flatten_tree(child, node["id"], depth + 1, out)


def attach_file_hierarchy(file_roots: dict[str, dict], parent_map: dict[str, str | None]) -> dict:
    """Build final tree with synthetic root and toctree-driven file parents."""
    synthetic_root = {
        "id": "pageindex_root",
        "kind": "root",
        "title": "Symfony Docs",
        "source": None,
        "breadcrumb": "Symfony Docs",
        "line_start": None,
        "line_end": None,
        "text": "",
        "summary": "Root node for Symfony documentation tree",
        "children": [],
    }

    for file_node in file_roots.values():
        file_node["_file_children"] = []

    for child_file, parent_file in parent_map.items():
        if child_file not in file_roots:
            continue
        if parent_file and parent_file in file_roots and parent_file != child_file:
            file_roots[parent_file]["_file_children"].append(file_roots[child_file])

    has_parent = {k: False for k in file_roots.keys()}
    for child_file, parent_file in parent_map.items():
        if child_file in has_parent and parent_file in file_roots and parent_file != child_file:
            has_parent[child_file] = True

    for file_name, file_node in file_roots.items():
        file_children = file_node.pop("_file_children", [])
        # toctree document children first, then section nodes
        file_node["children"] = file_children + file_node.get("children", [])
        if not has_parent.get(file_name, False):
            synthetic_root["children"].append(file_node)

    # Stable ordering
    synthetic_root["children"].sort(key=lambda x: x.get("source") or "")
    return synthetic_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PageIndex-style tree index from RST docs")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="Docs root directory")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument(
        "--summary-mode",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="How to create node summaries",
    )
    parser.add_argument("--summary-base-url", default=DEFAULT_SUMMARY_BASE_URL, help="OpenAI-compatible base URL for summaries")
    parser.add_argument("--summary-model", default=DEFAULT_SUMMARY_MODEL, help="Model name for LLM summaries")
    parser.add_argument("--summary-concurrency", type=int, default=8, help="Concurrent summary requests in LLM mode")
    parser.add_argument("--summary-input-chars", type=int, default=3200, help="Input text chars sent per summary request")
    parser.add_argument("--summary-output-chars", type=int, default=420, help="Max stored summary chars")
    args = parser.parse_args()

    docs_root = args.docs_dir.resolve()
    files = collect_rst_files(args.docs_dir)

    file_roots: dict[str, dict] = {}
    parent_map: dict[str, str | None] = {}

    with _progress() as progress:
        toctree_task = progress.add_task("Scanning toctrees", total=len(files))
        for file_path in files:
            rel = str(file_path.relative_to(args.docs_dir)).replace("\\", "/")
            parent_map.setdefault(rel, None)
            for link in parse_toctree_links(file_path, args.docs_dir):
                if link not in parent_map:
                    parent_map[link] = rel
            progress.advance(toctree_task)

        parse_task = progress.add_task("Parsing RST files", total=len(files))
        for file_path in files:
            rel = str(file_path.relative_to(args.docs_dir)).replace("\\", "/")

            raw = file_path.read_text(encoding="utf-8", errors="replace")
            resolved = resolve_includes(raw, file_path, docs_root, set())
            doctree = publish_doctree(
                resolved,
                source_path=str(file_path),
                settings_overrides={"report_level": 5, "halt_level": 5},
            )
            file_roots[rel] = build_doc_structure(doctree, rel)
            progress.advance(parse_task)

        full_tree = attach_file_hierarchy(file_roots, parent_map)

        flat_nodes: list[dict] = []
        flatten_tree(full_tree, parent_id=None, depth=0, out=flat_nodes)

        llm_summary_targets = 0
        if args.summary_mode == "llm":
            summary_total = len(
                [row for row in flat_nodes if row.get("kind") in {"section", "top"} and (row.get("text") or "").strip()]
            )
            summary_task = progress.add_task("Generating LLM summaries", total=summary_total)
            llm_summary_targets = asyncio.run(
                enrich_summaries_with_llm(
                    flat_nodes,
                    base_url=args.summary_base_url,
                    model=args.summary_model,
                    concurrency=args.summary_concurrency,
                    input_chars=args.summary_input_chars,
                    output_chars=args.summary_output_chars,
                    progress=progress,
                    task_id=summary_task,
                )
            )
            summary_by_id = {row["id"]: row.get("summary", "") for row in flat_nodes}
            _apply_summary_map(full_tree, summary_by_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tree_file = args.output_dir / "tree.json"
    nodes_file = args.output_dir / "nodes.jsonl"

    with open(tree_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "docs_dir": str(args.docs_dir),
                    "file_count": len(files),
                    "node_count": len(flat_nodes),
                    "summary_mode": args.summary_mode,
                    "summary_model": args.summary_model if args.summary_mode == "llm" else None,
                },
                "tree": full_tree,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(nodes_file, "w", encoding="utf-8") as f:
        for row in flat_nodes:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Built PageIndex tree: {tree_file}")
    print(f"Built PageIndex node index: {nodes_file}")
    print(f"Files: {len(files)}  Nodes: {len(flat_nodes)}")
    if args.summary_mode == "llm":
        print(f"LLM summaries requested for {llm_summary_targets} nodes")


if __name__ == "__main__":
    main()
