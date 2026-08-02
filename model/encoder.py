import torch
import math
import torch.nn as nn
from model.blocks import EncoderBlock
from model.blocks import model_config

class PositionalEncoding(nn.Module):

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(model_config.dropout)

        max_len = max(model_config.max_src_len, model_config.max_tar_len)
        embed_dim = model_config.embed_dim

        pe = torch.zeros(max_len, embed_dim)                                                # pe: one vector per pos
        pos = torch.arange(0, max_len).unsqueeze(1).float()                                 # (max_len, 1)

        # Sine/Cosine waves of diff. frequencies for each pos
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(pos * div_term)                                             # even dims - sin
        pe[:, 1::2] = torch.cos(pos * div_term)                                             # odd dims - cos

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, src_len, embed_dim)
        # adding pos. encoding to the actual seq_len
        x = x + self.pe[:, :x.size(1), :]                                                   
        return self.dropout(x)

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(model_config.vocab_size, model_config.embed_dim, padding_idx=model_config.pad_token_id)
        self.pos_encoding = PositionalEncoding()

        self.layers = nn.ModuleList([EncoderBlock() for _ in range(model_config.num_layers)])
        self.norm = nn.LayerNorm(model_config.embed_dim)
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:                             # tells attention to ignore PAD tokens

        # src: (batch, src_len)
        mask = (src != model_config.pad_token_id).unsqueeze(1).unsqueeze(2) 
        return mask                                                                         # (batch, 1, 1, src_len)
    
    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_mask = self.make_src_mask(src)
        x = self.embedding(src) * math.sqrt(model_config.embed_dim)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.norm(x)
        return x, src_mask                                                                  
