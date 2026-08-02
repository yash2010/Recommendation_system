import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import model_config

class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim:int = None, num_heads: int = None, droput: float = None):
        super().__init__()

        self.embed_dim = model_config.embed_dim
        self.num_heads = model_config.num_heads
        self.dropout = model_config.dropout

        self.head_dim = self.embed_dim // self.num_heads

        assert self.embed_dim % self.num_heads == 0, \
        f"embed_dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"

        # Learnable linear projections
        self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)        # query
        self.W_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)        # key
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)        # value
        self.W_o = nn.Linear(self.embed_dim, self.embed_dim, bias=False)        # output

        self.attn_dropout = nn.Dropout(self.dropout)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None,) -> torch.Tensor:
        
        """
        Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V , where d_k is dimension of heads 
        """
        
        # step 1: raw attention scores
        # Q: (batch, heads, seq, head_dim)
        # K.transpose: (batch, heads, head_dim, seq)
        # scores: (batch, heads, seq, seq) - one score per token pair
        scores = torch.matmul(Q, K.transpose(-2, -1))                           # Q @ K.T

        # step 2: scaled by sqrt(d_k) to prevent vanishing gradients
        scores = scores / math.sqrt(self.head_dim)
        
        # step 3: masking 
        # Mask is true for 2 situations:
        #       - padding mask = ignores PAD tokens
        #       - causal mask = decoder can't see future tokens
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        # step 4: softmax over last dimension
        # converts raw scores to probab. that sum to 1
        # -inf scores becomes 0 after softmax (masked pos. ignored)
        # weights: (batch, heads, seq, seq)
        weights = F.softmax(scores, dim= -1) 
        weights = self.attn_dropout(weights)

        # step 5: weighted sum of values
        # output: (batch, heads, seq, head_dim)
        output = torch.matmul(weights, V)

        return output, weights                 

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None,) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = query.shape[0]

        # step 1: project inputs inputs into Q, K, V
        # Each linear layer is transformed: (batch, seq, embed_dim)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # step 2: split into multiple heads
        # Reshape Q: (batch_size, seq, num_heads, head_dim) to (batch_size, heads, seq, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # attn_out: (batch, num_heads, seq, head_dim)
        attn_out, attn_w = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # step 4: concatanate the heads
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, -1, self.embed_dim)

        # step 5: linear projection
        # (batch, seq, embed_dim)
        output = self.W_o(attn_out)

        return output, attn_w
