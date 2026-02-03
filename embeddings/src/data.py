"""Document loading and preparation utilities."""

import json
from pathlib import Path


def load_wikipedia_abstracts(limit: int = 100000) -> list[dict]:
    """Load Wikipedia abstracts from HuggingFace datasets.

    Returns list of dicts with 'id', 'title', and 'text' fields.
    """
    from datasets import load_dataset

    docs = []
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    for i, row in enumerate(dataset):
        if i >= limit:
            break
        text = row["text"]
        first_para = text.split("\n\n")[0] if "\n\n" in text else text[:500]
        if len(first_para.strip()) < 50:
            continue
        docs.append(
            {
                "id": f"wiki-{i:06d}",
                "title": row["title"],
                "text": first_para.strip(),
            }
        )

    return docs


def save_documents(docs: list[dict], output_path: Path) -> Path:
    """Save documents to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(docs, f)
    return output_path


def load_documents(input_path: Path) -> list[dict]:
    """Load documents from JSON file."""
    with open(input_path) as f:
        return json.load(f)
