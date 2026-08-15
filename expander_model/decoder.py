import torch
import math
import torch.nn as nn
from expander_model.blocks import DecoderBlock
from expander_model.encoder import PositionalEncoding
from expander_model.config import model_config

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(model_config.vocab_size, model_config.embed_dim, padding_idx=model_config.pad_token_id)
        self.pos_encoding = PositionalEncoding()
        self.layers = nn.ModuleList([DecoderBlock() for _ in range(model_config.num_layers)])
        self.norm = nn.LayerNorm(model_config.embed_dim)
        self.output_proj = nn.Linear(model_config.embed_dim, model_config.vocab_size, bias = False)

    
    def make_tar_mask(self, tar: torch.Tensor) -> torch.Tensor:
        
        # tar: (batch, tar_len)
        tar_len = tar.shape[1]
        causal_mask = torch.tril(torch.ones(tar_len, tar_len, device=tar.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        pad_mask = (tar != model_config.pad_token_id)       # (batch, tar_len)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)       # (batch, 1, 1, tar_len)

        tar_mask = causal_mask & pad_mask
        return tar_mask
    
    def forward(self, tar: torch.Tensor, encoder_out: torch.Tensor, src_mask: torch.Tensor = None,) -> torch.Tensor:
        tar_mask = self.make_tar_mask(tar)
        x = self.embedding(tar) * math.sqrt(model_config.embed_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tar_mask)
        x = self.norm(x)
        logits = self.output_proj(x)                        # (batch, tgt_len, embed_dim) to (batch, tgt_len, vocab_size)
        return logits
        
