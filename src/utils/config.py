"""Config loader and helpers."""
from pathlib import Path
from typing import Any
import yaml


class Config(dict):
    """Dict that allows attribute access: cfg.training.batch_size instead of cfg['training']['batch_size']."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = Config(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    """Load YAML config from disk."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(data)


def save_config(config: Config, path: str | Path) -> None:
    """Save config to YAML (useful for snapshotting experiments)."""
    def _to_dict(obj):
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        return obj
    with open(path, "w") as f:
        yaml.safe_dump(_to_dict(config), f, default_flow_style=False, sort_keys=False)