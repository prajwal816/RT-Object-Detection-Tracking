"""YAML configuration loader with hierarchical merge support."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML configuration file.

    If *path* is ``None``, the built-in ``configs/default.yaml`` is loaded.

    Args:
        path: Absolute or relative path to a YAML file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}. Returning empty dict.")
        return {}

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    logger.info(f"Loaded config from [bold]{config_path}[/]")
    return cfg


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge *override* into *base* (override wins on conflict).

    Args:
        base: Base configuration dictionary.
        override: Override values.

    Returns:
        Merged configuration (new dict; inputs are not mutated).
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
