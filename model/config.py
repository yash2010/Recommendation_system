from dataclasses import dataclass

# Prompt 

SYSTEM_PROMPT = """You are helping build training data for a movie search system.
Given a vague movie query, expand it into a rich, specific description that would
help find relevant movies in a database of plot summaries.

Rules:
- Add specific themes, motifs, and atmosphere words
- Include genre terminology that appears in plot summaries
- Mention narrative elements (unreliable narrator, nonlinear structure, etc.)
- Keep it to 3-4 sentences
- Output ONLY the expanded query, no preamble, no explanation
- Never mention specific movie titles"""

@dataclass
class ModelConfig:

    vocab_size: int = 8000  # no. of unique tokens  
    max_src_len: int = 32   # max. input tokens
    max_tar_len: int = 128  # max. output tokens

    embed_dim: int = 256    # size of every token vector
    num_heads: int = 8      # no. of attention heads
    
    num_layers: int = 4     # no. of transformer blocks encoder + decoder
    ff_dim: int = 512       # hidden size of feedforward network (2x the embedded dimemsion)     
    
    dropout: float = 0.1    

    pad_token_id: int = 0   # padding - fills the short queries to the same length 
    sos_token_id: int = 1   # start of sentence
    eos_token_id: int = 2   # end of sentence
    unk_token_id: int = 3   # unknown token for the words which isn't included in the token

@dataclass
class TrainConfig:

    data_path: str = "data/training_pairs.json"
    train_split: float = 0.9

    epochs: int =  50
    batch_size:int = 8
    learning_rate: float = 3e-4
    warmup_steps: int = 100

    weight_decay: float = 0.01
    grad_clip: float = 1.0

    save_dir: str = "artifacts/expander"
    save_every: int = 10
    log_every: int = 5

@dataclass
class InferenceConfig:

    max_new_tokens: int = 128   # max. token to generate
    temperature: float = 0.7    # creativity: lower = more focused, higher = more creative
    top_k: int = 50             # samples from top 50 most similar/likely tokens

model_config = ModelConfig()
train_config = TrainConfig()
inference_config = InferenceConfig()



