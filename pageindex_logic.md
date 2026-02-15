# PageIndex Tree Build Logic

This document describes the logic used in `pageindex_build.py` to construct a hierarchical index of the Symfony documentation.

## Overview

The build process transforms flat RST files into a structured tree that preserves:

1.  **File-level hierarchy**: Derived from `.. toctree::` directives.
2.  **Internal document structure**: Derived from RST sections and titles.

**Input:** Source `.rst` files in `symfony-docs/`.
**Output:**

- `data/pageindex/tree.json`: The complete hierarchical tree.
- `data/pageindex/nodes.jsonl`: A flat list of all nodes for easy lookup.

## Build Steps

### 1. Scan Toctrees (File Hierarchy)

The script first scans all RST files to understand how they relate to each other.

- It looks for `.. toctree::` directives.
- It parses the file paths listed under these directives.
- It builds a `parent_map` where `child_file -> parent_file`.

### 2. Parse RST Files (internal Structure)

Each RST file is processed individually to extract its internal structure.

- **Resolution**: `resolve_includes` handles `.. include::` directives to create a single text stream.
- **Docutils Parsing**: The content is parsed into a Docutils document tree.
- **Node Extraction**:
  - **File Node**: The root node for the file.
  - **Top Node**: Represents content at the top of the file before the first section title.
  - **Section Nodes**: Formed by RST headers. These are nested recursively based on header levels.
  - **Content**: Text is cleaned and extracted for each section. Line numbers are recorded.

### 3. Attach File Hierarchy

The individual file trees are assembled into a global tree using the `parent_map` from Step 1.

- A **Synthetic Root** ("Symfony Docs") is created.
- If File A is a parent of File B (via toctree), File B's root node becomes a child of File A's root node.
- Files without parents are attached directly to the Synthetic Root.
- **Result**: A nested structure `Root -> File -> [Sections, Sub-Files]`.

### 4. Summarization

Nodes are enriched with summaries to assist in retrieval.

- **LLM Mode**: Sends text chunks to an LLM to generate concise summaries.

### 5. Flattening & Output

- The tree is traversed to create a flat list of nodes, linking children to parents via `parent_id`.
- The full nested tree is saved to `tree.json`.
- The flat list is saved to `nodes.jsonl`.

## Logic Diagram

```text
[ Start ]
    |
    v
[ Scan RST Files for Toctrees ]
    |
    v
[ Build Parent Map (Child -> Parent) ]
    |
    v
[ Parse Each RST File ]
    |
    +---> [ Resolve Includes ]
    |
    +---> [ Docutils Parse ]
    |
    +---> [ Extract Sections & Content ]
    |
    +---> [ Create Local File Tree ]
    |
    v
[ Attach File Hierarchy ]
    |
    v
[ Create Synthetic Root ]
    |
    v
[ Enrich Summaries (Heuristic or LLM) ]
    |
    v
[ Flatten Tree ]
    |
    +---> [ Save tree.json ]
    |
    +---> [ Save nodes.jsonl ]
    |
    v
[ End ]
```
