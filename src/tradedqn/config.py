"""Config loader — single source of truth for all tunable parameters (§7).

Loads ``config/config.yaml`` into an attribute-access ``Config`` namespace so
the rest of the codebase reads ``cfg.features.window_size`` instead of digging
through dicts (and so nothing is hardcoded in source).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = "config/config.yaml"


class Config(SimpleNamespace):
    """Attribute-access view over a config mapping (nested dicts become nested Configs)."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(**{key: _wrap(value) for key, value in data.items()})

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to plain dicts/lists (round-trips ``load_config``)."""
        return {key: _unwrap(value) for key, value in vars(self).items()}


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _unwrap(value: Any) -> Any:
    if isinstance(value, Config):
        return value.to_dict()
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


def resolve_path(path: str) -> Path:
    """Resolve ``path`` against the project root when it is relative."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


def assert_in_project(path: str) -> str:
    """§13 path-traversal guard: a *relative* path must resolve inside the project.

    Absolute paths pass through (so tmp dirs in tests work); relative paths that
    escape the project root (e.g. ``../../etc/x``) are refused.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    resolved = (_PROJECT_ROOT / candidate).resolve()
    if _PROJECT_ROOT not in resolved.parents and resolved != _PROJECT_ROOT:
        raise ValueError(f"relative path {path!r} resolves outside the project root")
    return str(resolved)


def load_config(path: str | None = None) -> Config:
    """Load a YAML config file into a :class:`Config`.

    Relative paths resolve from the project root, so the loader works regardless
    of the current working directory. Raises ``ValueError`` if the YAML root is
    not a mapping (a list or scalar config is always a mistake here).
    """
    config_path = resolve_path(path or DEFAULT_CONFIG_PATH)
    with open(config_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(
            f"config root must be a mapping, got {type(data).__name__} from {config_path}"
        )
    return Config(data)
