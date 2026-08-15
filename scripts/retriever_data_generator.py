"""
retriever_data_generator.py

Generates {"movies_id", "query", "plotsummary"} output to train the retrival model
Reads the database to acquire the data

Output:
    data/retriever_pairs.json
"""

import sys
import json
import csv
import argparse
import ollama
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYSTEM_PROMPT = """You are helping build training data for a movie search system.
Given a movie's title, genre, director, and plot summary, write short,
natural search queries that a real user might type looking for this movie.

Rules:
- Write exactly 3 queries, one per line, no numbering or bullets
- Each query should be a short, casual search phrase (5-12 words)
- Vary the angle across the 3: one about the plot/premise, one about the mood/vibe, one about cast or director
- Never use the movie's title
- Never copy exact phrases from the plot summary -- paraphrase in your own words
- Only reference settings, events, and characters that are explicitly stated
  in the plot summary given -- never invent details that aren't there
- Output ONLY the 3 queries. Do not write any introduction, label, or
  sentence like "Here are three queries" before them -- the very first
  line of your reply must already be the first query.

The example below shows the FORMAT and LENGTH you should match -- it has
nothing to do with the movie you're given. Never reuse its wording, even
if the movie you're given seems similar in genre or theme.

Format example:
tense courtroom drama about a lawyer hiding a secret
slow-burn story with a gray, melancholy mood
film starring an actress known for quiet, restrained performances
""".strip()

# Config

SOURCE_PATH: str = "data/tmdb_movies.csv"
OUTPUT_PATH: str = "data/retriever_pairs.json"
MODEL:str = "llama3.2:1b"
SAVE_EVERY: int = 10
QUERIES_PER_MOVIE: int = 3
PREAMBLE_MARKERS: tuple = ("here are", "sure", "certainly", "of course", "queries:", "query:")
EXAMPLE_QUERIES: set = {
    "tense courtroom drama about a lawyer hiding a secret",
    "slow-burn story with a gray, melancholy mood",
    "film starring an actress known for quiet, restrained performances",
}
MIN_QUERY_WORDS: int = 3
MAX_QUERY_WORDS: int = 14

def load_source_movies() -> list[dict]:
    with open(SOURCE_PATH, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_existing_pairs() -> list[dict]:
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"Resuming -- found {len(pairs)} existing pairs.")
        return pairs
    return []

def save_pairs(pairs: list[dict]) -> None:
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    
def generate_queries(row: dict, model: str = MODEL) -> list[str]:
    user_msg = (
        f"Title: {row['title']}\n"
        f"Genre: {row['genre']}\n"
        f"Director: {row['director']}\n"
        f"Cast: {row['cast']}\n"
        f"Plot: {row['plotsummary']}"
    )

    response = ollama.chat(
        model = model,
        messages = [
            {"role":"system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    text = response["message"]["content"].strip()

    lines = [line.strip("-*0123456789. ").strip() for line in text.split("\n")]
    queries = [
        line for line in lines
        if line
        and not line.endswith(":")
        and not any(marker in line.lower() for marker in PREAMBLE_MARKERS)
        and line.lower() not in EXAMPLE_QUERIES
        and MIN_QUERY_WORDS <= len(line.split()) <= MAX_QUERY_WORDS
    ]

    return queries[:QUERIES_PER_MOVIE]

def parse_args():
    parser = argparse.ArgumentParser(description="Generate retriever training pairs via Ollama")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process this many new movies this run then stop."
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

    movies = load_source_movies()
    print(f"Loaded {len(movies)} total movies from {SOURCE_PATH}")

    pairs = load_existing_pairs()
    already_done = set(p["movie_id"] for p in pairs)
    remaining = [row for row in movies if row["movie_id"] not in already_done]

    print(f"Already done: {len(already_done)}")
    print(f"Remaining: {len(remaining)}")

    if args.limit is not None:
        remaining = remaining[:args.limit]
        print(f"Batch limit applied, processing {len(remaining)} movies this run")

    print(f"using model: {args.model}\n")

    if not remaining:
        print ("Nothing to do, all the movies have been processed")
        return

    for i, row in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] {row['title']}")
        try:
            queries = generate_queries(row, model=args.model)
            for q in queries:
                pairs.append({
                    "movie_id": row["movie_id"],
                    "query": q,
                    "plotsummary": row["plotsummary"],
                })
            print(f"  -> {len(queries)} queries generated")
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        if i % SAVE_EVERY == 0:
            save_pairs(pairs)
            print(f"checkpoint saved -- {len(pairs)} total pairs so far")

    save_pairs(pairs)
    total_remaining_after = len(movies) - len(already_done) - len(remaining)
    print(f"\nDone this run. {len(pairs)} total pairs saved to {OUTPUT_PATH}")
    print(f"Remaining: {max(0, total_remaining_after)}")


if __name__ == "__main__":
    main()