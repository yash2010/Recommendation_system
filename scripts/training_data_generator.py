"""
generate_training_data.py

Generates (vague_query, expanded_query) training pairs using Ollama.
Reads queries from data/vague_queries.json.

Output:
    data/training_pairs.json
"""

import sys
import json
import argparse
import ollama
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import SYSTEM_PROMPT

# Config
QUERIES_PATH = "data/vague_queries.json"
OUTPUT_PATH  = "data/training_pairs.json"
MODEL        = "llama3.2:1b"     
SAVE_EVERY   = 10                

def expand_query(query: str, model: str = MODEL) -> str:

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Expand this vague movie query:\n{query}"}
        ]
    )
    return response["message"]["content"].strip()


def load_existing_pairs() -> list:

    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"Resuming -- found {len(pairs)} existing pairs")
        return pairs
    return []


def save_pairs(pairs: list) -> None:
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate training pairs via Ollama")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many NEW queries this run, then stop. "
             "Rerun the same command to continue from where you left off."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help=f"Ollama model to use (default: {MODEL})"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the full query list
    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        vague_queries = json.load(f)
    print(f"Loaded {len(vague_queries)} total queries from {QUERIES_PATH}")

    # Resume support -- skip queries already processed in earlier runs
    pairs = load_existing_pairs()
    already_done = set(p["input"] for p in pairs)
    remaining = [q for q in vague_queries if q not in already_done]

    print(f"Already done: {len(already_done)}")
    print(f"Remaining:    {len(remaining)}")

    # Apply batch limit if provided -- process only this many THIS run
    if args.limit is not None:
        remaining = remaining[:args.limit]
        print(f"Batch limit applied -- processing {len(remaining)} queries this run")

    print(f"Using model:  {args.model}\n")

    if not remaining:
        print("Nothing to do -- all queries already processed.")
        return

    for i, query in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] {query}")
        try:
            expanded = expand_query(query, model=args.model)
            pairs.append({"input": query, "target": expanded})
            print(f"  -> {expanded[:100]}...")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        # Checkpoint periodically so a disconnect doesn't lose progress
        if i % SAVE_EVERY == 0:
            save_pairs(pairs)
            print(f"  [checkpoint saved -- {len(pairs)} total pairs so far]")

    # Final save
    save_pairs(pairs)

    total_remaining_after = len(vague_queries) - len(pairs)
    print(f"\nDone this run. {len(pairs)} total pairs saved to {OUTPUT_PATH}")
    print(f"Remaining:    {max(0, total_remaining_after)}")


if __name__ == "__main__":
    main()
