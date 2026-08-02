import torch
from pathlib import Path
from expanders.base import BaseExpander
from model.model import QueryExpander
from model.tokenizer import Tokenizer

class LocalExpander(BaseExpander):

    def __init__(self, model_path:str = "artifacts/expander/checkpoint_best.pt", tokenizer_path:str = "artifacts/expander/tokenizer.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load(model_path, tokenizer_path)
    
    def _load(self, model_path: str, tokenizer_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run train.py first."
            )

        # Load tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.load(tokenizer_path)

        # Load model
        self.model = QueryExpander().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        print(f"LocalExpander loaded (epoch {checkpoint['epoch']}, "
              f"val_loss={checkpoint['val_loss']:.4f})")

    def expand(self, query: str) -> str:
        if not self.should_expand(query):
            return query

        try:
            expanded = self.model.generate(query, self.tokenizer)
            return expanded if expanded.strip() else query
        except Exception as e:
            print(f"Local expansion failed: {e}")
            return query   # fall back to original query
