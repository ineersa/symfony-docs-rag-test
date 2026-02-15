#!/usr/bin/env python3
"""Interactive retrieval REPL for PageIndex-style tree RAG."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from pageindex_common import TreeRetriever, load_nodes

DEFAULT_BASE_URL = "http://localhost:8052/v1"
DEFAULT_MODEL = "local-model"
DEFAULT_NODES_FILE = Path("data/pageindex/nodes.jsonl")


def _content_preview(text: str, max_chars: int) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return "(no extracted content)"
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def display_hits(console: Console, query: str, hits, top_k: int, content_chars: int, show_scores: bool) -> None:
    console.print()
    console.print(Panel(f"[bold]{query}[/]", title="Query", border_style="blue"))
    if not hits:
        console.print("[yellow]No results.[/]")
        return

    for i, h in enumerate(hits, 1):
        header = f"[bold cyan]#{i}[/]"
        if show_scores:
            header += f" [dim]score:[/] {h.score:.4f} [dim]dist:[/] {h.distance:.4f}"
        source = h.source
        if h.line_start is not None and h.line_end is not None:
            source += f":{h.line_start}-{h.line_end}"
        snippet = _content_preview(h.text, content_chars)
        body = f"[bold]{source}[/]\n[dim]{h.breadcrumb or ''}[/]\n\n{h.title}\n\n[dim]Content:[/] {snippet}"
        console.print(Panel(body, title=header, border_style="dim", padding=(0, 1)))

    console.print(f"[dim]Showing {min(len(hits), top_k)} results[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve from PageIndex-style tree")
    parser.add_argument("--nodes-file", type=Path, default=DEFAULT_NODES_FILE, help="Path to pageindex nodes.jsonl")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--beam-width", type=int, default=4, help="Nodes kept each depth")
    parser.add_argument("--max-depth", type=int, default=4, help="Max traversal depth")
    parser.add_argument("--content-chars", type=int, default=500, help="Max chars to show per result content")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM and use lexical fallback only")
    parser.add_argument("--llm-final-only", action="store_true", help="Use only LLM for final doc selection in LLM mode")
    parser.add_argument("--step-candidates", type=int, default=24, help="Max candidates sent to LLM per traversal step")
    parser.add_argument("--final-candidates", type=int, default=24, help="Max candidates sent to LLM for final ranking")
    parser.add_argument("--step-summary-chars", type=int, default=320, help="Summary chars per candidate in traversal")
    parser.add_argument("--final-summary-chars", type=int, default=420, help="Summary chars per candidate in final ranking")
    parser.add_argument("--step-text-chars", type=int, default=900, help="Body excerpt chars per candidate in traversal")
    parser.add_argument("--final-text-chars", type=int, default=1500, help="Body excerpt chars per candidate in final ranking")
    parser.add_argument("query", nargs="*", help="One-shot query")
    args = parser.parse_args()

    console = Console()
    if not args.nodes_file.is_file():
        raise SystemExit(f"Index file not found: {args.nodes_file}. Run pageindex_build.py first.")

    nodes = load_nodes(args.nodes_file)
    retriever = TreeRetriever(
        nodes,
        base_url=args.base_url,
        model=args.model,
        use_llm=not args.no_llm,
        llm_final_only=args.llm_final_only,
        step_candidate_limit=args.step_candidates,
        final_candidate_limit=args.final_candidates,
        step_summary_chars=args.step_summary_chars,
        final_summary_chars=args.final_summary_chars,
        step_text_chars=args.step_text_chars,
        final_text_chars=args.final_text_chars,
    )

    if args.query:
        query = " ".join(args.query)
        hits = retriever.retrieve(
            query,
            top_k=args.top_k,
            beam_width=args.beam_width,
            max_depth=args.max_depth,
        )
        display_hits(console, query, hits, args.top_k, args.content_chars, show_scores=args.no_llm)
        return

    console.print("[bold]PageIndex Retrieval REPL[/]")
    console.print("Commands: [dim]:quit, :top N, :beam N, :depth N, :help[/]\n")

    top_k = args.top_k
    beam = args.beam_width
    depth = args.max_depth

    while True:
        try:
            query = console.input("[bold green]query>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/]")
            break

        if not query:
            continue
        if query in {":quit", ":q"}:
            console.print("[dim]Bye![/]")
            break
        if query in {":help", ":h"}:
            console.print("[dim]Commands:[/]")
            console.print("  :quit / :q     - exit")
            console.print("  :top N         - set top-k")
            console.print("  :beam N        - set beam width")
            console.print("  :depth N       - set max depth")
            continue
        if query.startswith(":top "):
            try:
                top_k = int(query.split()[1])
                console.print(f"[dim]top_k={top_k}[/]")
            except Exception:
                console.print("[red]Usage: :top N[/]")
            continue
        if query.startswith(":beam "):
            try:
                beam = int(query.split()[1])
                console.print(f"[dim]beam={beam}[/]")
            except Exception:
                console.print("[red]Usage: :beam N[/]")
            continue
        if query.startswith(":depth "):
            try:
                depth = int(query.split()[1])
                console.print(f"[dim]depth={depth}[/]")
            except Exception:
                console.print("[red]Usage: :depth N[/]")
            continue

        hits = retriever.retrieve(query, top_k=top_k, beam_width=beam, max_depth=depth)
        display_hits(console, query, hits, top_k, args.content_chars, show_scores=args.no_llm)


if __name__ == "__main__":
    main()
