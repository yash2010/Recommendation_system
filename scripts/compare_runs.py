"""
Usage:
    python compare_runs.py                     # compare ALL runs found
    python compare_runs.py --last 3             # compare only the 3 most recent
    python compare_runs.py --filter bpe          # only runs with "bpe" in the name
"""

import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

RUNS_DIR    = Path("artifacts/expander/runs")
OUTPUT_PATH = "artifacts/expander/comparison.png"

# One fixed color per tokenizer so runs are visually grouped by tokenizer,
# not by run order.
TOKENIZER_COLORS = {
    "word_level":  "#2E75B6",
    "bpe_library": "#C0392B",
    "bpe_scratch": "#2E7D32",
}
FALLBACK_COLOR = "#8E44AD"


def parse_args():
    parser = argparse.ArgumentParser(description="Compare training runs")
    parser.add_argument("--last", type=int, default=None,
                         help="Only compare the N most recent runs")
    parser.add_argument("--filter", type=str, default=None,
                         help="Only include runs whose folder name contains this text")
    return parser.parse_args()


def discover_runs(filter_text: str = None) -> list[Path]:

    if not RUNS_DIR.exists():
        print(f"No runs directory found at {RUNS_DIR}")
        return []

    run_dirs = sorted(
        [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "history.json").exists()]
    )

    if filter_text:
        run_dirs = [d for d in run_dirs if filter_text in d.name]

    return run_dirs


def load_history(run_dir: Path) -> list[dict]:
    with open(run_dir / "history.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_tokenizer(run_dir: Path) -> str:
   
    meta_path = run_dir / "config_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            tokenizer = json.load(f).get("tokenizer")
        if tokenizer:
            return tokenizer

    for name in TOKENIZER_COLORS:
        if run_dir.name.endswith(name):
            return name
    return "unknown"


def main():
    args = parse_args()

    run_dirs = discover_runs(args.filter)

    if args.last:
        run_dirs = run_dirs[-args.last:]

    if not run_dirs:
        print("No runs found to compare.")
        return

    print(f"Comparing {len(run_dirs)} runs:")
    for d in run_dirs:
        print(f"  - {d.name}")
    print()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    summary_rows = []

    for run_dir in run_dirs:
        history = load_history(run_dir)
        if not history or "train_loss" not in history[0]:
            print(f"  Skipping {run_dir.name} — incomplete history")
            continue

        epochs       = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses   = [h["val_loss"] for h in history]

        # Shorten the label -- drop the long timestamp, keep it readable
        label = run_dir.name

        tokenizer = get_tokenizer(run_dir)
        color = TOKENIZER_COLORS.get(tokenizer, FALLBACK_COLOR)
        ax1.plot(epochs, train_losses, label=label, color=color, linewidth=2)
        ax2.plot(epochs, val_losses, label=label, color=color, linewidth=2)

        best_val = min(val_losses)
        best_epoch = epochs[val_losses.index(best_val)]
        summary_rows.append((label, train_losses[-1], best_val, best_epoch))

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Train Loss")
    ax1.set_title("Training Loss Comparison"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Comparison"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved comparison plot to {OUTPUT_PATH}")

    print("\n" + "=" * 90)
    print(f"{'Run':<40} {'Final Train Loss':<18} {'Best Val Loss':<16} {'Best Epoch'}")
    print("=" * 90)
    for label, final_train, best_val, best_epoch in summary_rows:
        print(f"{label:<40} {final_train:<18.4f} {best_val:<16.4f} {best_epoch}")


if __name__ == "__main__":
    main()