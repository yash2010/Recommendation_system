import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.config import model_config

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(model_config.embed_dim, model_config.ff_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
            nn.Linear(model_config.ff_dim, model_config.embed_dim),
            nn.Dropout(model_config.dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Encoder
class EncoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention()
        self.ff = FeedForward()

        self.norm1 = nn.LayerNorm(model_config.embed_dim)
        self.norm2 = nn.LayerNorm(model_config.embed_dim)
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        
        # self attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, mask)  # self attention: Q=K=V=x
        x = residual + self.dropout(attn_out)

        # feed forward with residual
        residual = x
        x = self.norm2(x)
        ff_out = self.ff(x)
        x = residual + self.dropout(ff_out)

        return x
    

# Decoder
class DecoderBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.masked_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.ff = FeedForward()

        self.norm1 = nn.LayerNorm(model_config.embed_dim)
        self.norm2 = nn.LayerNorm(model_config.embed_dim)
        self.norm3 = nn.LayerNorm(model_config.embed_dim)
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, src_mask: torch.Tensor = None, tar_mask: torch.Tensor = None) -> torch.Tensor:

        # masked self attention
        # decoder attends to its own previous tokens
        # tar_mask prevents it into looking at future tokens
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.masked_attn(x, x, x, tar_mask)
        x = residual + self.dropout(attn_out)

        # cross-attention - bridge btw encoder and decoder
        # query comes from decoder
        # key and value comes from encoder
        residual = x
        x = self.norm2(x)
        attn_out, _ = self.masked_attn(x, encoder_out, encoder_out, src_mask)
        x = residual + self.dropout(attn_out)

        # adds non-linearity
        # feed forward
        residual = x
        x = self.norm3(x)
        ff_out = self.ff(x)
        x = residual + self.dropout(ff_out)

        return x 