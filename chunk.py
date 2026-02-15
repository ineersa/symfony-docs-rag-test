#!/usr/bin/env python3
"""RST-aware chunker for Symfony documentation.

Walks symfony-docs/, resolves .. include:: directives inline,
parses with docutils, splits by section hierarchy, and outputs
chunks as JSON Lines to data/chunks.jsonl.
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from docutils.core import publish_doctree
from docutils import nodes
from docutils.parsers.rst import roles, directives, Directive

# ── Register Sphinx-specific roles/directives as no-ops ────────────
# docutils doesn't know about Sphinx roles like :ref:, :doc:, :class:, etc.
# Register them as generic roles so they're parsed as inline text, not errors.
_SPHINX_ROLES = [
    "ref", "doc", "class", "func", "meth", "attr", "mod", "obj", "exc",
    "data", "const", "term", "command", "option", "envvar", "file",
    "samp", "guilabel", "menuselection", "kbd", "mailheader", "mimetype",
    "newsgroup", "program", "regexp", "manpage", "dfn", "abbr",
    "index", "download", "hoverxref", "numref", "eq", "math",
    "any", "pep", "rfc", "confval", "rst:role", "rst:dir",
    "phpclass", "phpmethod", "phpfunction", "method",
]


def _generic_role(role, rawtext, text, lineno, inliner, options=None, content=None):
    """Generic role handler: just render the text as inline literal."""
    node = nodes.literal(rawtext, text)
    return [node], []


for _role_name in _SPHINX_ROLES:
    roles.register_local_role(_role_name, _generic_role)

# Register common Sphinx directives as no-ops (parsed as comments)
_SPHINX_DIRECTIVES = [
    "toctree", "versionadded", "versionchanged", "deprecated",
    "seealso", "todo", "index", "glossary", "only",
    "best-practice", "screencast",
    "sidebar", "rst-class", "caution",
]


class _NoOpDirective(Directive):
    """Directive that captures its content and renders it as a block quote."""
    required_arguments = 0
    optional_arguments = 100
    has_content = True
    option_spec = {}
    final_argument_whitespace = True

    def run(self):
        # Render the content as a literal block so it's preserved
        text = "\n".join(self.content) if self.content else ""
        if text.strip():
            node = nodes.literal_block(text, text)
            return [node]
        return []


for _dir_name in _SPHINX_DIRECTIVES:
    directives.register_directive(_dir_name, _NoOpDirective)


# ── configuration-block: split nested code-blocks by language ──────
_CODE_BLOCK_SPLIT_RE = re.compile(r"^\.\.\s+code-block::\s*(\S+)", re.MULTILINE)


class _ConfigBlockDirective(Directive):
    """Parses configuration-block into separate literal_block nodes per language."""
    required_arguments = 0
    optional_arguments = 0
    has_content = True
    option_spec = {}
    final_argument_whitespace = True

    def run(self):
        if not self.content:
            return []

        raw = "\n".join(self.content)
        parts = _CODE_BLOCK_SPLIT_RE.split(raw)

        result_nodes: list[nodes.Node] = []

        # parts[0] = text before the first code-block (usually blank)
        if parts[0].strip():
            text = parts[0].strip()
            result_nodes.append(nodes.literal_block(text, text))

        # Alternating: lang, content, lang, content, ...
        for i in range(1, len(parts), 2):
            lang = parts[i]
            code = parts[i + 1] if i + 1 < len(parts) else ""
            # Dedent: the code was indented under .. code-block::
            lines = code.split("\n")
            non_empty = [l for l in lines if l.strip()]
            if non_empty:
                min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
                code = "\n".join(l[min_indent:] for l in lines)
            code = code.strip()
            if code:
                node = nodes.literal_block(code, code)
                node["language"] = lang
                result_nodes.append(node)

        return result_nodes


directives.register_directive("configuration-block", _ConfigBlockDirective)

# ── Config ──────────────────────────────────────────────────────────
DOCS_DIR = Path("symfony-docs")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"
EXCLUDE_DIRS = {"_build", "_images", ".github"}
TARGET_CHUNK_SIZE = 1000  # characters
MAX_CHUNK_SIZE = 1000    # hard cap — force-split at \n boundaries
OVERLAP_SIZE = 200  # characters for fallback splits
MIN_CHUNK_SIZE = 50  # skip chunks shorter than this


# ── Include resolution ──────────────────────────────────────────────
_INCLUDE_RE = re.compile(r"^\.\.\s+include::\s+(.+)$", re.MULTILINE)


def resolve_includes(text: str, file_path: Path, docs_root: Path, resolved: set[str]) -> str:
    """Recursively replace ``.. include::`` directives with file contents."""

    def _replace(match: re.Match) -> str:
        ref = match.group(1).strip()
        # Resolve relative to docs root if starts with /, else relative to file
        if ref.startswith("/"):
            inc_path = docs_root / ref.lstrip("/")
        else:
            inc_path = file_path.parent / ref
        inc_path = inc_path.resolve()

        if not inc_path.is_file():
            return match.group(0)  # leave directive as-is if file missing

        resolved.add(str(inc_path))
        inc_text = inc_path.read_text(encoding="utf-8", errors="replace")
        # Recurse in case the included file also has includes
        return resolve_includes(inc_text, inc_path, docs_root, resolved)

    return _INCLUDE_RE.sub(_replace, text)


# ── Doctree → text extraction ──────────────────────────────────────
def _render_table_row(row: nodes.row) -> str:
    """Render a single table row as: cell1 | cell2 | ..."""
    cells: list[str] = []
    for entry in row.children:
        if isinstance(entry, nodes.entry):
            # Flatten entry content to a single line
            text = entry.astext().replace("\n", " ").strip()
            cells.append(text)
    return " | ".join(cells)


def node_to_text(node: nodes.Node) -> str:
    """Convert a docutils node subtree to plain-text, preserving code blocks."""
    parts: list[str] = []

    # Skip system messages (docutils error/warning nodes)
    if isinstance(node, nodes.system_message):
        return ""
    elif isinstance(node, nodes.table):
        # Render tables row-by-row with blank lines between rows
        for tgroup in node.findall(nodes.tgroup):
            for tbody in tgroup.findall(nodes.tbody):
                for row in tbody.children:
                    if isinstance(row, nodes.row):
                        row_text = _render_table_row(row)
                        if row_text:
                            parts.append(row_text)
            # Also include thead if present
            for thead in tgroup.findall(nodes.thead):
                header_parts: list[str] = []
                for row in thead.children:
                    if isinstance(row, nodes.row):
                        row_text = _render_table_row(row)
                        if row_text:
                            header_parts.append(row_text)
                if header_parts:
                    parts = header_parts + parts
    elif isinstance(node, nodes.literal_block):
        # Preserve code blocks as-is with fencing
        lang = node.get("language", "")
        code = node.astext()
        if lang:
            parts.append(f"```{lang}\n{code}\n```")
        else:
            parts.append(f"```\n{code}\n```")
    elif isinstance(node, nodes.Text):
        parts.append(node.astext())
    elif isinstance(node, nodes.title):
        parts.append(node.astext())
    elif isinstance(node, nodes.reference):
        text = node.astext()
        uri = node.get("refuri", "")
        if uri:
            parts.append(f"{text} ({uri})")
        else:
            parts.append(text)
    elif isinstance(node, nodes.section):
        # Sections handled at a higher level; skip recursive descent here
        pass
    elif isinstance(node, nodes.problematic):
        # Problematic nodes contain unparsed RST (e.g. Sphinx roles)
        # Extract the raw text content
        parts.append(node.astext())
    else:
        for child in node.children:
            parts.append(node_to_text(child))

    return "\n\n".join(p for p in parts if p.strip())


def extract_section_content(section: nodes.section) -> str:
    """Extract text from a section's direct content (not nested sub-sections)."""
    parts: list[str] = []
    for child in section.children:
        if isinstance(child, nodes.section):
            continue  # Skip nested sections, they become their own chunks
        parts.append(node_to_text(child))
    return "\n\n".join(p for p in parts if p.strip())


def node_line_range(node: nodes.Node) -> tuple[int | None, int | None]:
    """Return min/max source line numbers from node subtree."""
    lines: list[int] = []
    own = getattr(node, "line", None)
    if isinstance(own, int):
        lines.append(own)

    for child in node.findall():
        ln = getattr(child, "line", None)
        if isinstance(ln, int):
            lines.append(ln)

    if not lines:
        return None, None
    return min(lines), max(lines)


def get_section_title(section: nodes.section) -> str:
    """Get section title text."""
    for child in section.children:
        if isinstance(child, nodes.title):
            return child.astext()
    return ""


# ── Chunking logic ─────────────────────────────────────────────────
_CODE_FENCE_RE = re.compile(r"(```[^\n]*\n.*?\n```)", re.DOTALL)


def split_by_code_blocks(text: str, target: int) -> list[str]:
    """Split text at code-fence boundaries.

    Each code block becomes its own segment.  Small adjacent prose/code
    segments are merged together up to *target* characters.
    """
    segments = _CODE_FENCE_RE.split(text)
    segments = [s.strip() for s in segments if s.strip()]

    if len(segments) <= 1:
        return segments  # nothing to split

    merged: list[str] = []
    current = ""

    for seg in segments:
        candidate = f"{current}\n\n{seg}".strip() if current else seg
        if len(candidate) > target and current:
            merged.append(current.strip())
            current = seg
        else:
            current = candidate

    if current.strip():
        merged.append(current.strip())

    return merged if merged else [text]


def split_with_overlap(text: str, target: int, overlap: int) -> list[str]:
    """Split text at paragraph boundaries with overlap."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > target and current:
            chunks.append(current.strip())
            # Overlap: take the tail of the current chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = f"{overlap_text}\n\n{para}".strip()
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def force_split_lines(text: str, hard_cap: int) -> list[str]:
    """Last-resort split at \n boundaries when text exceeds *hard_cap*."""
    if len(text) <= hard_cap:
        return [text]

    lines = text.split("\n")
    chunks: list[str] = []
    current = ""

    for line in lines:
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > hard_cap and current:
            chunks.append(current.strip())
            current = line
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]


def split_chunk(text: str) -> list[str]:
    """Full splitting pipeline: code blocks → paragraphs → lines (hard cap)."""
    # Stage 1: split on code-fence boundaries
    stage1 = split_by_code_blocks(text, TARGET_CHUNK_SIZE)

    # Stage 2: split remaining large segments at paragraph boundaries
    stage2: list[str] = []
    for seg in stage1:
        if len(seg) > TARGET_CHUNK_SIZE:
            stage2.extend(split_with_overlap(seg, TARGET_CHUNK_SIZE, OVERLAP_SIZE))
        else:
            stage2.append(seg)

    # Stage 3: hard cap — force-split at \n boundaries
    stage3: list[str] = []
    for seg in stage2:
        if len(seg) > MAX_CHUNK_SIZE:
            stage3.extend(force_split_lines(seg, MAX_CHUNK_SIZE))
        else:
            stage3.append(seg)

    return stage3


def merge_tiny_chunks(chunks: list[dict], min_size: int, max_size: int) -> list[dict]:
    """Merge chunks smaller than *min_size* into their neighbours.

    Pass 1 – try to append a tiny chunk to the **previous** chunk.
    Pass 2 – try to prepend any remaining tiny chunk to the **next** chunk.
    Merging is only done if the combined text stays within *max_size*.
    """
    if len(chunks) <= 1:
        return chunks

    def _can_merge(a: dict, b: dict) -> bool:
        a_meta = a.get("metadata", {})
        b_meta = b.get("metadata", {})
        return a_meta.get("source") == b_meta.get("source")

    def _merge_metadata(a: dict, b: dict) -> None:
        a_meta = a.get("metadata", {})
        b_meta = b.get("metadata", {})

        a_start = a_meta.get("line_start")
        b_start = b_meta.get("line_start")
        a_end = a_meta.get("line_end")
        b_end = b_meta.get("line_end")

        starts = [v for v in (a_start, b_start) if isinstance(v, int)]
        ends = [v for v in (a_end, b_end) if isinstance(v, int)]

        if starts:
            a_meta["line_start"] = min(starts)
        if ends:
            a_meta["line_end"] = max(ends)

    # Pass 1: merge backward
    backward: list[dict] = []
    for chunk in chunks:
        if len(chunk["text"]) < min_size and backward:
            prev = backward[-1]
            combined = prev["text"] + "\n\n" + chunk["text"]
            if len(combined) <= max_size and _can_merge(prev, chunk):
                prev["text"] = combined
                _merge_metadata(prev, chunk)
                continue
        backward.append(chunk)

    # Pass 2: merge forward
    forward: list[dict] = []
    skip = False
    for i, chunk in enumerate(backward):
        if skip:
            skip = False
            continue
        if len(chunk["text"]) < min_size and i + 1 < len(backward):
            nxt = backward[i + 1]
            combined = chunk["text"] + "\n\n" + nxt["text"]
            if len(combined) <= max_size and _can_merge(chunk, nxt):
                nxt["text"] = combined
                _merge_metadata(nxt, chunk)
                continue  # tiny chunk absorbed into next; next will be added normally
        forward.append(chunk)

    return forward


def chunk_document(doctree: nodes.document, source_file: str) -> list[dict]:
    """Extract chunks from a parsed doctree."""
    chunks: list[dict] = []
    chunk_counter = 0

    def _process_section(section: nodes.section, breadcrumb: list[str]):
        nonlocal chunk_counter

        title = get_section_title(section)
        current_breadcrumb = breadcrumb + [title] if title else breadcrumb
        breadcrumb_str = " > ".join(current_breadcrumb)

        section_children = [
            child for child in section.children if not isinstance(child, nodes.section)
        ]
        section_ranges = [node_line_range(child) for child in section_children]
        section_starts = [start for start, _ in section_ranges if isinstance(start, int)]
        section_ends = [end for _, end in section_ranges if isinstance(end, int)]
        line_start = min(section_starts) if section_starts else None
        line_end = max(section_ends) if section_ends else None

        # Get this section's own content (excluding nested sections)
        content = extract_section_content(section)

        if content.strip():
            if len(content) <= TARGET_CHUNK_SIZE:
                # Single chunk
                chunk_id = hashlib.md5(
                    f"{source_file}:{breadcrumb_str}:{chunk_counter}".encode()
                ).hexdigest()
                chunks.append({
                    "id": chunk_id,
                    "text": content.strip(),
                    "metadata": {
                        "source": source_file,
                        "breadcrumb": breadcrumb_str,
                        **({"line_start": line_start} if line_start is not None else {}),
                        **({"line_end": line_end} if line_end is not None else {}),
                    },
                })
                chunk_counter += 1
            else:
                # Full splitting pipeline: code blocks → paragraphs → hard cap
                sub_chunks = split_chunk(content)

                for i, sub in enumerate(sub_chunks):
                    chunk_id = hashlib.md5(
                        f"{source_file}:{breadcrumb_str}:{chunk_counter}:{i}".encode()
                    ).hexdigest()
                    chunks.append({
                        "id": chunk_id,
                        "text": sub.strip(),
                        "metadata": {
                            "source": source_file,
                            "breadcrumb": breadcrumb_str,
                            "part": i + 1,
                            **({"line_start": line_start} if line_start is not None else {}),
                            **({"line_end": line_end} if line_end is not None else {}),
                        },
                    })
                    chunk_counter += 1

        # Recurse into nested sections
        for child in section.children:
            if isinstance(child, nodes.section):
                _process_section(child, current_breadcrumb)

    # Handle top-level content (before any section)
    top_content_parts: list[str] = []
    top_nodes: list[nodes.Node] = []
    for child in doctree.children:
        if isinstance(child, nodes.section):
            _process_section(child, [])
        else:
            text = node_to_text(child)
            if text.strip():
                top_content_parts.append(text)
                top_nodes.append(child)

    top_ranges = [node_line_range(node) for node in top_nodes]
    top_starts = [start for start, _ in top_ranges if isinstance(start, int)]
    top_ends = [end for _, end in top_ranges if isinstance(end, int)]
    top_line_start = min(top_starts) if top_starts else None
    top_line_end = max(top_ends) if top_ends else None

    if top_content_parts:
        top_content = "\n\n".join(top_content_parts)
        if len(top_content) <= TARGET_CHUNK_SIZE:
            chunk_id = hashlib.md5(f"{source_file}:top:{chunk_counter}".encode()).hexdigest()
            chunks.append({
                "id": chunk_id,
                "text": top_content.strip(),
                "metadata": {
                    "source": source_file,
                    "breadcrumb": "(top-level)",
                    **({"line_start": top_line_start} if top_line_start is not None else {}),
                    **({"line_end": top_line_end} if top_line_end is not None else {}),
                },
            })
        else:
            # Full splitting pipeline: code blocks → paragraphs → hard cap
            sub_chunks = split_chunk(top_content)

            for i, sub in enumerate(sub_chunks):
                chunk_id = hashlib.md5(
                    f"{source_file}:top:{chunk_counter}:{i}".encode()
                ).hexdigest()
                chunks.append({
                    "id": chunk_id,
                    "text": sub.strip(),
                    "metadata": {
                        "source": source_file,
                        "breadcrumb": "(top-level)",
                        "part": i + 1,
                        **({"line_start": top_line_start} if top_line_start is not None else {}),
                        **({"line_end": top_line_end} if top_line_end is not None else {}),
                    },
                })
                chunk_counter += 1

    # Merge tiny chunks into neighbours instead of discarding them
    chunks = merge_tiny_chunks(chunks, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)

    return chunks


def _char_pos_to_line_number(text: str, pos: int) -> int:
    """Convert 0-based character position to 1-based line number."""
    if pos <= 0:
        return 1
    return text.count("\n", 0, pos) + 1


def add_line_spans(chunks: list[dict], full_text: str) -> None:
    """Annotate chunk metadata with line_start/line_end based on full text."""
    if not chunks or not full_text:
        return

    cursor = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue

        idx = full_text.find(text, cursor)
        if idx == -1:
            idx = full_text.find(text)
            if idx == -1:
                continue

        line_start = _char_pos_to_line_number(full_text, idx)
        line_end = _char_pos_to_line_number(full_text, idx + len(text))

        meta = chunk.setdefault("metadata", {})
        meta["line_start"] = line_start
        meta["line_end"] = line_end

        cursor = idx + len(text)


# ── File collection ────────────────────────────────────────────────
def collect_rst_files(docs_dir: Path) -> list[Path]:
    """Collect all .rst and .rst.inc files, excluding blacklisted dirs."""
    files: list[Path] = []
    for root, dirs, filenames in os.walk(docs_dir):
        # Prune excluded directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(".rst") or fn.endswith(".rst.inc"):
                files.append(Path(root) / fn)
    return files


# ── Main ───────────────────────────────────────────────────────────
def main():
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    docs_root = DOCS_DIR.resolve()

    console.print(f"[bold]Scanning[/] {DOCS_DIR} for RST files...")
    all_files = collect_rst_files(DOCS_DIR)
    console.print(f"Found [cyan]{len(all_files)}[/] RST files")

    # First pass: scan all files for includes to build the resolved set
    resolved_includes: set[str] = set()

    # Process files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    errors: list[str] = []

    with Progress(console=console) as progress:
        task = progress.add_task("Chunking...", total=len(all_files))

        for filepath in all_files:
            progress.update(task, advance=1, description=f"[cyan]{filepath.name}[/]")

            # Skip if this file was already inlined via .. include::
            if str(filepath.resolve()) in resolved_includes:
                continue

            try:
                raw_text = filepath.read_text(encoding="utf-8", errors="replace")

                # Resolve includes
                text = resolve_includes(raw_text, filepath, docs_root, resolved_includes)

                # Parse with docutils (suppress warnings)
                settings_overrides = {
                    "report_level": 5,  # suppress all warnings
                    "halt_level": 5,
                }
                doctree = publish_doctree(
                    text,
                    source_path=str(filepath),
                    settings_overrides=settings_overrides,
                )

                # Convert relative path for metadata
                rel_path = str(filepath.relative_to(DOCS_DIR))
                file_chunks = chunk_document(doctree, rel_path)
                add_line_spans(file_chunks, text)
                all_chunks.extend(file_chunks)

            except Exception as e:
                errors.append(f"{filepath}: {e}")

    # Merge tiny chunks into neighbours (replaces old filter)
    before_count = len(all_chunks)
    all_chunks = merge_tiny_chunks(all_chunks, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
    merged_count = before_count - len(all_chunks)

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    console.print()
    console.print(f"[bold green]✓[/] Wrote [cyan]{len(all_chunks)}[/] chunks to [bold]{OUTPUT_FILE}[/]")
    console.print(f"  Skipped [yellow]{len(resolved_includes)}[/] files (inlined via include)")
    console.print(f"  Merged [yellow]{merged_count}[/] tiny chunks (< {MIN_CHUNK_SIZE} chars) into neighbours")

    if errors:
        console.print(f"\n[bold red]⚠ {len(errors)} errors:[/]")
        for err in errors[:10]:
            console.print(f"  [red]{err}[/]")

    # Quick stats
    sizes = [len(c["text"]) for c in all_chunks]
    if sizes:
        console.print(f"\n[bold]Chunk stats:[/]")
        console.print(f"  Total:   {len(sizes)}")
        console.print(f"  Avg:     {sum(sizes) // len(sizes)} chars")
        console.print(f"  Min:     {min(sizes)} chars")
        console.print(f"  Max:     {max(sizes)} chars")
        console.print(f"  Median:  {sorted(sizes)[len(sizes) // 2]} chars")


if __name__ == "__main__":
    main()
