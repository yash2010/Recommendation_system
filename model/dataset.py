import json
import torch 
from torch.utils.data import Dataset, DataLoader
from model.config import model_config, train_config

class MovieQueryDataset(Dataset):

    def __init__(self, pairs: list[dict], tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        # Vague query encoding (no SOS, no EOS)
        src = self.tokenizer.encode(pair["input"], max_len = model_config.max_src_len, add_sos = False, add_eos = False)

        # target input encoding (SOS + expanded_query, no EOS) 
        tar_in = self.tokenizer.encode(pair["target"], max_len = model_config.max_tar_len, add_sos = True, add_eos = False)

        # target output decoding (expanded_query + EOS, no SOS)
        tar_out = self.tokenizer.encode(pair["target"], max_len = model_config.max_tar_len, add_sos = False, add_eos = True)

        return{
            "src": torch.tensor(src, dtype=torch.long),
            "tar_in": torch.tensor(tar_in, dtype=torch.long),
            "tar_out": torch.tensor(tar_out, dtype=torch.long)
        }

def build_dataloaders(TokenizerClass, data_path: str = None, batch_size: int = None,) -> tuple:

    if data_path is None: data_path = train_config.data_path
    if batch_size is None: batch_size = train_config.batch_size

    with open(data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs from {data_path}")

    all_txts = [p["input"] for p in pairs] + [p["target"] for p in pairs]
    tokenizer = TokenizerClass()
    tokenizer.build(all_txts)
    tokenizer.save("artifacts/expander/tokenizer.json")

    split = int(len(pairs) * train_config.train_split)
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    train_dataset = MovieQueryDataset(train_pairs, tokenizer)
    val_dataset = MovieQueryDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer
