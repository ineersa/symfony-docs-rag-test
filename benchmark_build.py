#!/usr/bin/env python3
"""Build retrieval benchmark questions from Symfony RST files.

Generates 2-3 questions per file and stores:
- question text
- source file
- gold answer line span
- supporting quote
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from chunk import DOCS_DIR, collect_rst_files, resolve_includes

DEFAULT_OUTPUT = Path("data/benchmark/questions.jsonl")
DEFAULT_BASE_URL = "http://localhost:8052/v1"
DEFAULT_MODEL = "local-model"

ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*?(?:Exception|Interface|Trait|Bundle|Component|Kernel|Command|Event)\b")


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

    normalized_text = re.sub(r"\s+", " ", text).strip()
    normalized_quote = re.sub(r"\s+", " ", quote).strip()
    if not normalized_text or not normalized_quote:
        return None

    n_idx = normalized_text.find(normalized_quote)
    if n_idx == -1:
        return None

    # Approximate start/end by searching near the normalized prefix.
    prefix = normalized_text[:n_idx]
    prefix_words = prefix.split(" ")
    anchor = " ".join(prefix_words[-6:]).strip()
    anchor_idx = text.find(anchor) if anchor else 0
    if anchor_idx == -1:
        anchor_idx = 0
    fallback_idx = text.find(quote[: min(60, len(quote))], anchor_idx)
    if fallback_idx == -1:
        fallback_idx = anchor_idx

    start = _line_number_from_pos(text, fallback_idx)
    end = _line_number_from_pos(text, min(len(text), fallback_idx + len(quote)))
    return start, end


def candidate_blocks(text: str) -> list[str]:
    blocks = text.split("\n\n")
    out: list[str] = []
    in_code = False

    for raw in blocks:
        block = raw.strip()
        if not block:
            continue

        if block.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if block.startswith(".. "):
            continue
        if set(block) in ({"="}, {"-"}, {"~"}, {"^"}):
            continue
        out.append(block)

    return out


def sentence_candidates(blocks: list[str]) -> list[str]:
    out: list[str] = []
    for block in blocks:
        for match in re.finditer(r"[^.!?]{40,420}[.!?]", block, re.DOTALL):
            sentence = match.group(0).strip()
            if len(sentence) >= 90:
                out.append(sentence)
    return out


def choose_entity(sentences: list[str], text: str) -> tuple[str | None, str | None]:
    entities = ENTITY_RE.findall(text)
    if not entities:
        return None, None
    for entity in entities:
        for sentence in sentences:
            if entity in sentence:
                return entity, sentence
    return entities[0], None


def _preview(text: str, max_chars: int = 140) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    return one_line[:max_chars].rstrip()


def heuristic_questions(source: str, text: str) -> list[dict]:
    blocks = candidate_blocks(text)
    sentences = sentence_candidates(blocks)
    if not sentences:
        return []

    questions: list[dict] = []
    entity, entity_sentence = choose_entity(sentences, text)

    if entity and entity_sentence:
        questions.append(
            {
                "question": f"In `{source}`, what behavior or recommendation is described for `{entity}`?",
                "answer_quote": entity_sentence,
                "kind": "entity",
                "required_entity": entity,
                "difficulty": "hard",
            }
        )

    for sentence in sentences:
        if len(questions) >= 3:
            break
        if entity and sentence == entity_sentence:
            continue
        question = f"Which Symfony guidance matches this scenario: {_preview(sentence)}...?"
        questions.append(
            {
                "question": question,
                "answer_quote": sentence,
                "kind": "general",
                "required_entity": None,
                "difficulty": "hard",
            }
        )

    return questions[:3]


def _extract_json(text: str) -> list[dict] | None:
    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
    try:
        value = json.loads(payload)
        if isinstance(value, list):
            return value
    except json.JSONDecodeError:
        pass
    return None


def llm_questions(client: OpenAI, model: str, source: str, text: str) -> list[dict]:
    excerpt = text[:7000]
    prompt = (
        "Generate exactly 3 tricky retrieval benchmark questions from this Symfony docs file. "
        "At least one question must mention a concrete class, exception, interface, trait, bundle, or component name. "
        "Return ONLY JSON array with items: question, answer_quote, kind(entity|general), required_entity(optional), difficulty. "
        "answer_quote must be exact contiguous text from the excerpt.\n\n"
        f"Source file: {source}\n"
        "Excerpt:\n"
        f"{excerpt}"
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
    )
    data = _extract_json(response.output_text)
    if not data:
        return []
    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not item.get("question") or not item.get("answer_quote"):
            continue
        out.append(
            {
                "question": str(item.get("question", "")).strip(),
                "answer_quote": str(item.get("answer_quote", "")).strip(),
                "kind": str(item.get("kind", "general")).strip() or "general",
                "required_entity": item.get("required_entity"),
                "difficulty": str(item.get("difficulty", "hard")).strip() or "hard",
            }
        )
    return out[:3]


def llm_questions_with_retry(
    client: OpenAI,
    model: str,
    source: str,
    text: str,
    max_retries: int,
) -> list[dict]:
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            return llm_questions(client, model, source, text)
        except (APIConnectionError, APITimeoutError):
            if attempt == attempts:
                raise
            time.sleep(min(2 * attempt, 8))
    return []


def question_id(source_file: str, question: str, line_start: int, line_end: int) -> str:
    raw = f"{source_file}|{question}|{line_start}|{line_end}"
    return hashlib.md5(raw.encode()).hexdigest()


def load_existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = item.get("id")
            if isinstance(qid, str):
                out.add(qid)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark questions from Symfony docs")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="Path to Symfony docs root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--generator", choices=["heuristic", "llm"], default="heuristic", help="Question generator mode")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL for LLM mode")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name for LLM mode")
    parser.add_argument("--limit", type=int, default=0, help="Max files to process (0 = all)")
    parser.add_argument("--resume", action="store_true", help="Append to existing output and skip duplicate question IDs")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shards")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="LLM request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for transient LLM connection failures")
    parser.add_argument(
        "--on-llm-error",
        choices=["fallback", "fail"],
        default="fallback",
        help="Fallback to heuristic or stop when LLM call fails",
    )
    args = parser.parse_args()

    console = Console()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_files = collect_rst_files(args.docs_dir)
    docs_root = args.docs_dir.resolve()

    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("Invalid shard configuration")

    selected = [p for i, p in enumerate(all_files) if i % args.shard_count == args.shard_index]
    if args.limit > 0:
        selected = selected[: args.limit]

    existing_ids = load_existing_ids(args.output) if args.resume else set()
    write_mode = "a" if args.resume else "w"

    client = None
    if args.generator == "llm":
        client = OpenAI(base_url=args.base_url, api_key="not-needed", timeout=args.request_timeout)

    generated = 0
    skipped = 0
    timestamp = datetime.now(timezone.utc).isoformat()

    with open(args.output, write_mode, encoding="utf-8") as out, Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating questions...", total=len(selected))
        for file_path in selected:
            progress.update(task, description=f"[cyan]{file_path.name}[/]  g={generated} s={skipped}")
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            resolved = resolve_includes(raw, file_path, docs_root, set())
            rel = str(file_path.relative_to(args.docs_dir))

            if args.generator == "llm" and client is not None:
                try:
                    candidates = llm_questions_with_retry(
                        client, args.model, rel, resolved, args.max_retries
                    )
                except (APIConnectionError, APITimeoutError) as exc:
                    if args.on_llm_error == "fail":
                        raise
                    console.print(
                        f"[yellow]LLM request failed for[/] {rel}: {exc}. "
                        "Falling back to heuristic generation."
                    )
                    candidates = heuristic_questions(rel, resolved)
                if not candidates:
                    candidates = heuristic_questions(rel, resolved)
            else:
                candidates = heuristic_questions(rel, resolved)

            for item in candidates:
                span = find_line_span(resolved, item["answer_quote"])
                if not span:
                    skipped += 1
                    continue
                line_start, line_end = span
                qid = question_id(rel, item["question"], line_start, line_end)
                if qid in existing_ids:
                    skipped += 1
                    continue

                record = {
                    "id": qid,
                    "question": item["question"],
                    "difficulty": item.get("difficulty", "hard"),
                    "kind": item.get("kind", "general"),
                    "required_entity": item.get("required_entity"),
                    "source_file": rel,
                    "answer_line_start": line_start,
                    "answer_line_end": line_end,
                    "answer_quote": item["answer_quote"],
                    "generator": {
                        "mode": args.generator,
                        "model": args.model if args.generator == "llm" else "heuristic",
                        "base_url": args.base_url if args.generator == "llm" else None,
                        "timestamp": timestamp,
                        "shard_index": args.shard_index,
                        "shard_count": args.shard_count,
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_ids.add(qid)
                generated += 1

            progress.advance(task)

    console.print(f"[green]Generated[/] {generated} questions")
    console.print(f"[yellow]Skipped[/]   {skipped} items (duplicates or unresolved spans)")
    console.print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
