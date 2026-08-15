import tempfile
import json
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from expander_model.config import model_config, tokenizer_config, SPECIAL_TOKENS, SPECIAL_TOKEN_IDS

class Tokenizer():

    def __init__(self):
        self._tok = None
        self.vocab_size = 0
        self.pad_id = model_config.pad_token_id
        self.sos_id = model_config.sos_token_id
        self.eos_id = model_config.eos_token_id
        self.unk_id = model_config.unk_token_id
    
    def build(self, texts: list[str], max_vocab: int = None) -> None:

        if max_vocab is None:
            max_vocab = model_config.vocab_size
        
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", errors="replace") as f:
            for text in texts:
                f.write(text.strip() + "\n")
            corpus_path = f.name
        
        min_frequency = getattr(tokenizer_config, "min_frequency", 2)

        self._tok = ByteLevelBPETokenizer()
        self._tok.train(files=[corpus_path], vocab_size=max_vocab, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS)

        Path(corpus_path).unlink(missing_ok = True)  # Delete the temporary file

        for token, expected_id in SPECIAL_TOKEN_IDS.items():
            actual_id = self._tok.token_to_id(token)
            if actual_id != expected_id:
                raise RuntimeError(
                    f"Special token {token} got id {actual_id} from the tokenizer, "
                    f"but expander_train.yaml's model config expects {expected_id}. "
                    f"Check pad/sos/eos/unk_token_id in config/expander_train.yaml."
                )

        self.vocab_size = self._tok.get_vocab_size()
        print(f"Tokenizer built with vocab size: {self.vocab_size}")

    # Coverts the each words to IDs 
    def encode(self, text:str, max_len:int = None, add_sos:bool = False, add_eos:bool = False) ->list[int]:
        
        if self._tok is None:
            raise ValueError("Tokenizer has not been built or loaded yet.")

        encoding = self._tok.encode(text.strip())
        ids = encoding.ids

        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        if max_len is not None:
            if len(ids) < max_len:
                ids = ids + [self.pad_id] * (max_len - len(ids))
            else:
                ids = ids[:max_len]
        
        return ids
    
    # Converts list of IDs to words
    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if self._tok is None:
            raise ValueError("Tokenizer has not been built or loaded yet.")

        special = {self.pad_id, self.sos_id, self.eos_id}
        keep_ids = []

        for id_ in ids:
            if id_ == self.eos_id:
                break
            if skip_special and id_ in special:
                continue
            keep_ids.append(id_)

        return self._tok.decode(keep_ids)
    
    def save(self, path: str) -> None:

        out_dir = Path(path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = Path(path).stem

        self._tok.save_model(str(out_dir), prefix)

        manifest = {
            "vocab_file": f"{prefix}-vocab.json",
            "merges_file": f"{prefix}-merges.txt",
            "vocab_size": self.vocab_size,
        }

        with open(path, "w", encoding = "utf-8") as f:
            json.dump(manifest, f, indent = 2)
        
        print(f"Tokenizer saved to {path} ({self.vocab_size} tokens)")

    
    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    
        out_dir = Path(path).parent
        vocab_file = out_dir / manifest["vocab_file"]
        merges_file = out_dir / manifest["merges_file"]

        self._tok = ByteLevelBPETokenizer(str(vocab_file), str(merges_file))
        self.vocab_size = manifest["vocab_size"]

        print(f"Tokenizer loaded from {path} ({self.vocab_size} tokens)")
    
    def __len__(self) -> int:
        return self.vocab_size


