import re
import json
from pathlib import Path
from collections import Counter
from model.config import model_config

class Tokenizer():

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: dict[str, int] = {}
        self.vocab_size: int = 0

        self.pad_id = model_config.pad_token_id
        self.sos_id = model_config.sos_token_id
        self.eos_id = model_config.eos_token_id
        self.unk_id = model_config.unk_token_id
    

    def _clean(self, text:str) -> str:

        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", " ", text)        # removes punctuations
        text = re.sub(r"\s+", " ", text)                # removes multiple spaces
        return text
    
    def _tokenize(self, text:str) -> list[str]:
        return self._clean(text).split()                # splits the text into tokens
    
    def build(self, texts: list[str], max_vocab: int = None) -> None:

        if max_vocab is None:
            max_vocab = model_config.vocab_size
        
        counter = Counter()                             # counts the word frequencies across all texts
        for text in texts:
            tokens = self._tokenize(text)
            counter.update(tokens)
        
        print(f"Unique words found: {len(counter)}")

        self.token2id = {                               # reverves IDs for the special cases
            "<PAD>": self.pad_id,
            "<SOS>": self.sos_id,
            "<EOS>": self.eos_id,
            "<UNK>": self.unk_id
        }

        for word, _ in counter.most_common(max_vocab - 4):  # most frequent words are assigned Id's starting from 4
            self.token2id[word] = len(self.token2id)
        
        self.id2token = {id_: tok for tok, id_ in self.token2id.items()}   # mirror of token2id is id2token
        self.vocab_size = len(self.token2id)

        print(f"Vocabulary size: {self.vocab_size}")
    
    # Coverts the each words to IDs 
    def encode(self, text:str, max_len:int = None, add_sos:bool = False, add_eos:bool = False) ->list[int]:

        tokens = self._tokenize(text)
        ids = []
        for tok in tokens:
            id_ = self.token2id.get(tok, self.unk_id)
            ids.append(id_)

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
    def decode(self, ids: list[str], skip_special: bool = True) -> str:

        special = {self.pad_id, self.sos_id, self.eos_id}
        tokens = []

        for id_ in ids:
            if skip_special and id_ in special:
                continue

            if id_ == self.eos_id:
                break
            tokens.append(self.id2token.get(id_, "<UNK>"))
        
        return " ".join(tokens)
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, indent= 2, ensure_ascii=False)
        print(f"Tokenizer saved to {path} ({self.vocab_size} tokens)")
    
    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.token2id = json.load(f)
        self.id2token = {id_: tok for tok, id_ in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        print(f"Tokenizer loaded from {path} ({self.vocab_size} tokens)")
    
    def __len__(self) -> int:
        return self.vocab_size


