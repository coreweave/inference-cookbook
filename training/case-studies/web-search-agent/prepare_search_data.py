# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Write the search agent's training questions as Gym data, for grpo_search_agent.yaml.

SimpleQA (the default) has short fact-seeking questions that a search agent answers in a few tool
calls, so a GRPO group gets mixed rewards. BrowseComp's questions take dozens of tool calls; the
Gym eval recipe measures the model on them. Both are OpenAI's published test sets. Each row gets
the prompt and tools of Gym's benchmarks/browsecomp/prepare.py (Gym v0.4.0, Apache-2.0), so that
training prompts match the benchmark's.

The launcher runs this before training (``PREPARE_DATA`` in run.env.example). It uses only the
standard library.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import urllib.request
from collections.abc import Iterable, Mapping
from datetime import date
from itertools import islice
from pathlib import Path
from typing import Any

# dataset -> (CSV URL, published number of rows)
DATASETS = {
    "simpleqa": (
        "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
        4326,
    ),
    "browsecomp": (
        "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv",
        1266,
    ),
}
OUTPUT = Path("/work/hotload/nemo-gym/search.jsonl")  # grpo_search_agent.yaml's data_path
AGENT = "browsecomp_agent"  # the agent in grpo_search_agent.yaml
SYSTEM_PROMPT = (
    "You are a General Agent. Today's date: {date}. "
    "Your mission is to leverage a diverse set of tools to help the user conduct "
    "an in-depth investigation of their question, continuously reflect, and "
    "ultimately deliver a precise answer.\n\n"
    "Throughout the investigation, strictly observe the following principles:\n"
    "1. Whenever you encounter uncertain information, proactively invoke search "
    "tools to verify it.\n"
    "2. You can only invoke one tool in each round.\n"
    "3. Prioritize high-credibility sources (authoritative websites, academic "
    "databases, professional media) and maintain a critical stance toward "
    "low-credibility ones. Cite the source of any information you use with "
    "a format [^index^].\n"
    "4. You should not respond to the user with a counter-question, but instead "
    "do your best to provide an accurate answer.\n"
    "5. When providing the final answer, begin by explaining the reasoning "
    "process. Avoid presenting only the final answer, as this makes it "
    "difficult to understand."
)
QUERY_SUFFIX = (
    "\n\nYour response should be in the following format:\n"
    "Explanation: {your explanation for your final answer}\n"
    "Exact Answer: {your succinct, final answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)
TOOLS = [
    {
        "type": "function",
        "name": "search",
        "description": (
            "Web Search API, works like Google Search. All queries will be searched in "
            "parallel. If you want to search with multiple keywords, put them in a single query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search queries. All queries are executed in parallel.",
                }
            },
            "required": ["queries"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "browse",
        "description": (
            "Visit specific webpage(s) and return their full text content. Use this to read "
            "the complete content of web pages found during search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "URL(s) of webpage(s) to visit.",
                },
                "goal": {
                    "type": "string",
                    "description": "What specific information you are looking for.",
                },
            },
            "required": ["urls"],
        },
        "strict": False,
    },
]


def decrypt(ciphertext: str, password: str) -> str:
    """Decrypt one BrowseComp field: base64 XORed with the SHA-256 of the row's canary."""
    data = base64.b64decode(ciphertext)
    digest = hashlib.sha256(password.encode()).digest()
    key = digest * (len(data) // len(digest)) + digest[: len(data) % len(digest)]
    return bytes(a ^ b for a, b in zip(data, key, strict=True)).decode()


def download(dataset: str) -> list[dict[str, str]]:
    """Download a dataset's rows.

    Raises:
        ValueError: The download does not have the published number of rows.
    """
    url, expected = DATASETS[dataset]
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode()
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != expected:
        raise ValueError(f"expected {expected} {dataset} rows, got {len(rows)}")
    return rows


def example(question: str, answer: str, today: date | None = None) -> dict[str, Any]:
    """A Gym example for the BrowseComp agent: its prompt and tools, the question and answer."""
    system = SYSTEM_PROMPT.format(date=(today or date.today()).isoformat())
    return {
        "agent_ref": {"name": AGENT},
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": question + QUERY_SUFFIX},
            ],
            "tools": copy.deepcopy(TOOLS),
        },
        "question": question,
        "ground_truth": answer,
    }


def row_example(dataset: str, row: Mapping[str, str], today: date | None = None) -> dict[str, Any]:
    """The Gym example for one dataset row; BrowseComp's question and answer are encrypted."""
    if dataset == "browsecomp":
        return example(
            decrypt(row["problem"], row["canary"]), decrypt(row["answer"], row["canary"]), today
        )
    return example(row["problem"], row["answer"], today)


def write(examples: Iterable[Mapping[str, Any]], path: Path) -> int:
    """Write examples as JSON lines, replacing ``path`` only once all are written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    count = 0
    try:
        with temporary.open("w") as output:
            for item in examples:
                output.write(json.dumps(item) + "\n")
                count += 1
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def main(argv: list[str] | None = None, rows: list[dict[str, str]] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="simpleqa")
    parser.add_argument("--max-samples", type=int, help="first N questions (default: all)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    selected = islice(rows if rows is not None else download(args.dataset), args.max_samples)
    count = write((row_example(args.dataset, row) for row in selected), args.output)
    print(f"wrote {count} {args.dataset} questions to {args.output}")


if __name__ == "__main__":
    main()
