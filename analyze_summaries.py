
import json
from pathlib import Path

nodes_file = Path("data/pageindex/nodes.jsonl")

lengths = []
missing_count = 0

if not nodes_file.exists():
    print(f"File not found: {nodes_file}")
    exit(1)

with open(nodes_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            summary = data.get("summary")
            if summary is None:
                missing_count += 1
            else:
                lengths.append(len(summary))
        except json.JSONDecodeError:
            continue

if not lengths:
    print("No summaries found.")
else:
    min_len = min(lengths)
    max_len = max(lengths)
    avg_len = sum(lengths) / len(lengths)
    print(f"Count: {len(lengths)}")
    print(f"Missing/Null: {missing_count}")
    print(f"Smallest: {min_len}")
    print(f"Largest: {max_len}")
    print(f"Average: {avg_len:.2f}")
