"""Configuration loading and management."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    merged = {}
    for config in configs:
        merged.update(config)
    return merged


def get_config_path(config_name: str) -> str:
    """Get full path to config file."""
    return str(Path(__file__).parent.parent.parent / "configs" / f"{config_name}.yaml")
