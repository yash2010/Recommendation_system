import torch
import torch.nn as nn
from expander_model.encoder import Encoder
from expander_model.decoder import Decoder
from expander_model.tokenizers.bpe_library import Tokenizer
from expander_model.config import model_config, inference_config

class QueryExpander(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self._init_weights()
    
    def _init_weights(self):
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
    
    def forward(self, src: torch.Tensor, tar_in: torch.Tensor,) -> torch.Tensor:

        encoder_out, src_mask = self.encoder(src)
        logits = self.decoder(tar_in, encoder_out, src_mask)
        return logits
    
    @torch.no_grad()
    def generate(self, query: str, tokenizer: Tokenizer, max_new_tokens: int = None, temperature: float = None, top_k: int = None,) -> str:

        if max_new_tokens is None: max_new_tokens = inference_config.max_new_tokens
        if temperature is None: temperature = inference_config.temperature
        if top_k is None: top_k = inference_config.top_k

        self.eval()

        device = next(self.parameters()).device

        # Encode src
        src_ids = tokenizer.encode(query, max_len=model_config.max_src_len, add_sos=False, add_eos=False)
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        encoder_out, src_mask = self.encoder(src)

        # Decode
        generated = [model_config.sos_token_id]
        for _ in range(max_new_tokens):
            
            # build current tar_seq
            tar = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
            
            # get logits for all positions
            logits = self.decoder(tar, encoder_out, src_mask)

            # take logits at last pos. (next token)
            nxt_token_logits = logits[0, -1, :]                 # vocab_size

            # apply temp.
            nxt_token_logits = nxt_token_logits / temperature

            # top_k
            if top_k > 0:
                top_k_values, _ = torch.topk(nxt_token_logits, top_k)
                min_top_k = top_k_values[-1]
                nxt_token_logits = nxt_token_logits.masked_fill(nxt_token_logits < min_top_k, float("-inf"))
            
            # sample next token
            probs = torch.softmax(nxt_token_logits, dim=-1)
            nxt_token = torch.multinomial(probs, num_samples=1).item()

            # stop if EOS
            if nxt_token == model_config.eos_token_id:
                break
            
            generated.append(nxt_token)
        # Decode generated ID to text
        return tokenizer.decode(generated[1:])
        
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
