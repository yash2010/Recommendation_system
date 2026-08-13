import os
import yaml
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


_raw = _load_yaml()

tmdb_config     = _to_namespace(_raw["tmdb"])
pipeline_config = _to_namespace(_raw["pipeline"])

tmdb_config.token = os.environ.get("TMDB_TOKEN")
