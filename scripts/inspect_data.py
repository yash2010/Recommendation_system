import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.config import get_tokenizer_class

DATA_PATH = Path("data/training_pairs.json")


def percentile(sorted_vals: list, pct: float):
    idx = min(int(len(sorted_vals) * pct), len(sorted_vals) - 1)
    return sorted_vals[idx]


def report(name: str, token_counts: list) -> None:
    counts = sorted(token_counts)
    n = len(counts)
    print(f"{name}: min={counts[0]} max={counts[-1]} avg={sum(counts)/n:.1f} "
          f"median={percentile(counts, 0.5)} p95={percentile(counts, 0.95)} p99={percentile(counts, 0.99)}")


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    inputs  = [p["input"] for p in pairs]
    targets = [p["target"] for p in pairs]

    TokenizerClass = get_tokenizer_class()
    tokenizer = TokenizerClass()
    tokenizer.build(inputs + targets)

    total_words, total_tokens = 0, 0
    src_token_counts, tar_token_counts = [], []

    for text, bucket in [(t, src_token_counts) for t in inputs] + [(t, tar_token_counts) for t in targets]:
        ids = tokenizer.encode(text, max_len=None, add_sos=False, add_eos=False)
        total_words  += len(text.split())
        total_tokens += len(ids)
        bucket.append(len(ids))

    print(f"Pairs: {len(pairs)}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Tokens per word: {total_tokens / total_words:.2f}")
    print()
    report("Input tokens ", src_token_counts)
    report("Target tokens", tar_token_counts)


if __name__ == "__main__":
    main()
