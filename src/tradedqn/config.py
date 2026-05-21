"""Config loader — single source of truth for all tunable parameters (§7).

Loads ``config/config.yaml`` into an attribute-access ``Config`` namespace so
the rest of the codebase reads ``cfg.features.window_size`` instead of digging
through dicts (and so nothing is hardcoded in source).
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = "config/config.yaml"
SUPPORTED_CONFIG_MAJOR = 1  # §8.1 — config-version compatibility checked at load


class Config(SimpleNamespace):
    """Attribute-access view over a config mapping (nested dicts become nested Configs)."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(**{key: _wrap(value) for key, value in data.items()})

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to plain dicts/lists (round-trips ``load_config``)."""
        return {key: _unwrap(value) for key, value in vars(self).items()}


def _wrap(value: Any) -> Any:
    """Recursively wrap dicts as ``Config`` and lists element-wise."""
    if isinstance(value, dict):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(item) for item in value]
    return value


def _unwrap(value: Any) -> Any:
    """Recursively convert ``Config``/lists back to plain dicts/lists."""
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
    requested = path or DEFAULT_CONFIG_PATH
    config_path = resolve_path(requested)
    if config_path.exists():  # source checkout (the normal `uv run` path)
        text = config_path.read_text(encoding="utf-8")
    else:  # §14 — installed wheel: fall back to the packaged config
        text = _packaged_config_text(requested, config_path)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(
            f"config root must be a mapping, got {type(data).__name__} from {config_path}"
        )
    _check_version(data, config_path)
    return Config(data)


def _packaged_config_text(requested: str, checkout_path: Path) -> str:
    """§14 — read ``config.yaml`` shipped inside the installed ``tradedqn`` wheel.

    When the source checkout has no config file (the project was pip-installed,
    not run from the repo), load the copy bundled in the package; otherwise raise
    a clear, actionable error.
    """
    if Path(requested).name == DEFAULT_CONFIG_PATH.split("/")[-1]:
        resource = importlib.resources.files("tradedqn") / "config.yaml"
        if resource.is_file():
            return resource.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"config not found at {checkout_path} — run TradeDQN from the project "
        "checkout (config/config.yaml lives at the repo root) or pass an explicit path"
    )


def _check_version(data: dict[str, Any], config_path: Path) -> None:
    """§8.1 — refuse a config whose major version this code can't support."""
    version = data.get("version")
    if version is None:
        raise ValueError(f"config {config_path} is missing the required 'version' key")
    major = str(version).split(".")[0]
    if major != str(SUPPORTED_CONFIG_MAJOR):
        raise ValueError(
            f"config version {version!r} is incompatible — this build supports "
            f"major version {SUPPORTED_CONFIG_MAJOR}.x"
        )
