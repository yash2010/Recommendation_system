import json
import torch
from pathlib import Path
from expanders.base import BaseExpander
from model.model import QueryExpander
from model.config import get_tokenizer_class, RUNS_DIR


class LocalExpander(BaseExpander):

    def __init__(self, run_dir: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if run_dir is None:
            run_dir = self._find_best_run()

        model_path     = str(Path(run_dir) / "checkpoint_best.pt")
        tokenizer_path = str(Path(run_dir) / "tokenizer.json")

        self._load(model_path, tokenizer_path, run_dir)

    def _find_best_run(self) -> str:
     
        runs_dir = Path(RUNS_DIR)
        if not runs_dir.exists():
            raise FileNotFoundError(f"No runs directory found at {RUNS_DIR}")

        modelsTrained = []
        for d in runs_dir.iterdir():
            if not d.is_dir() or not (d / "checkpoint_best.pt").exists():
                continue
            metadata_path = d / "config_metadata.json"
            if not metadata_path.exists():
                continue
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            modelsTrained.append((metadata["best_val_loss"], d))

        if not modelsTrained:
            raise FileNotFoundError(f"No completed runs with recorded val_loss found in {RUNS_DIR}")

        best_val_loss, best_dir = min(modelsTrained, key=lambda pair: pair[0])
        print(f"Auto-selected best run: {best_dir.name} (val_loss={best_val_loss:.4f})")
        return str(best_dir)

    def _read_tokenizer_name(self, run_dir: str) -> str:
        
        metadata_path = Path(run_dir) / "config_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata["tokenizer"]

    def _load(self, model_path: str, tokenizer_path: str, run_dir: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run train.py first."
            )

        tokenizer_name = self._read_tokenizer_name(run_dir)
        TokenizerClass = get_tokenizer_class(tokenizer_name)
        self.tokenizer = TokenizerClass()
        self.tokenizer.load(tokenizer_path)

        self.model = QueryExpander().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(f"LocalExpander loaded (epoch {checkpoint['epoch']}, "
              f"val_loss={checkpoint['val_loss']:.4f}, tokenizer={tokenizer_name})")

    def expand(self, query: str) -> str:
        if not self.should_expand(query):
            return query

        try:
            expanded = self.model.generate(query, self.tokenizer)
            return expanded if expanded.strip() else query
        except Exception as e:
            print(f"Local expansion failed: {e}")
            return query