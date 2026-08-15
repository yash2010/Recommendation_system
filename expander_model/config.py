import yaml
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_YAML_PATH = PROJECT_ROOT/"config"/"train.yaml"
API_YAML_PATH = PROJECT_ROOT/"config"/"api.yaml"



def _load_yaml(path:Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _to_namespace(d:dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns

# param imports form traim.yaml
_train_yaml = _load_yaml(TRAIN_YAML_PATH)

model_config = _to_namespace(_train_yaml["model"])
train_config = _to_namespace(_train_yaml["train"])
inference_config = _to_namespace(_train_yaml["inference"])
RUNS_DIR = _train_yaml["runs_dir"]

_active_tokenizer_name = _train_yaml["tokenizer"]
_tokenizer_block = _train_yaml["tokenizers"][_active_tokenizer_name]
 
tokenizer_config = _to_namespace(_tokenizer_block)
tokenizer_config.name = _active_tokenizer_name
tokenizer_config.class_ = _tokenizer_block["class"]

# Single source of truth for BPE tokenizers: symbol -> the id it must land on,
# derived from model_config so it can never drift out of sync with train.yaml.
SPECIAL_TOKEN_IDS = {
    "<PAD>": model_config.pad_token_id,
    "<SOS>": model_config.sos_token_id,
    "<EOS>": model_config.eos_token_id,
    "<UNK>": model_config.unk_token_id,
}
# Ordered by id, since HuggingFace's tokenizers library assigns special-token
# ids sequentially in the order this list is given.
SPECIAL_TOKENS = sorted(SPECIAL_TOKEN_IDS, key=SPECIAL_TOKEN_IDS.get)


def get_tokenizer_class(name: str = None):
    """
    Resolve a tokenizer class by name (module + class from train.yaml's
    `tokenizers:` block). Defaults to whichever tokenizer is currently
    active (`tokenizer:` in train.yaml) when no name is given.
    """
    import importlib

    if name is None:
        name = tokenizer_config.name

    block = _train_yaml["tokenizers"][name]
    module = importlib.import_module(block["module"])
    return getattr(module, block["class"])

# param imports from api.yaml
_api_yaml = _load_yaml(API_YAML_PATH)

api_config = _to_namespace(_api_yaml["api"])
database_config = _to_namespace(_api_yaml["database"])
expander_config = _to_namespace(_api_yaml["expander"])
SYSTEM_PROMPT = _api_yaml["system_prompt"].strip()

