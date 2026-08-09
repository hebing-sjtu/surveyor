"""Writing the settings the app collects from the user.

``config.py`` reads, and never writes a line itself. Everything that puts a value
on disk lives here, so there is one place that knows the file formats:
``config.toml`` for ordinary settings, ``.env`` for secrets, and a pointer file
under the user's config directory recording which library folder to open.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any

import tomllib

from .config import Settings, app_config_dir, get_settings, reload_settings

CONFIG_FILE = "config.toml"
ENV_FILE = ".env"

_ENV_HEADER = "# Secrets for Surveyor. Never commit this file."
# What a key with nothing in it yet looks like: an invitation to type between the
# quotes, rather than a bare name that reads like a mistake.
_BLANK = '""'

# What a fresh `.env` offers to fill in. The bot credentials are the ones that
# cannot be typed into the app — Settings collects the model's API key itself —
# so a blank library gives no clue that they exist unless the file names them.
_BOT_SECRETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Feishu / Lark app, for `surveyor feishu-connect`", ("FEISHU_APP_ID", "FEISHU_APP_SECRET")),
    ("WeCom 智能机器人, for `surveyor wecom-connect`", ("WECOM_BOT_ID", "WECOM_BOT_SECRET")),
)

_HEADER = """\
# Surveyor configuration.
#
# Written by the app, and safe to edit by hand. Secrets never live here: this
# file names the environment variables to read them from, and the values
# themselves belong in .env next to it.
"""


def set_library_root(path: Path | str) -> Path:
    """Remember which folder to treat as the library, and create it."""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    ensure_env_file(resolved)
    pointer = app_config_dir() / "home"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(str(resolved), encoding="utf-8")
    return resolved


def forget_library_root() -> None:
    pointer = app_config_dir() / "home"
    pointer.unlink(missing_ok=True)


# --------------------------------------------------------------------- .env


def read_env_file(root: Path) -> dict[str, str]:
    path = root / ENV_FILE
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.strip().strip("'\"")
    return values


def ensure_env_file(root: Path) -> Path:
    """Make sure ``.env`` exists and names the bot keys, blank, ready to fill in.

    Bot credentials are the one thing the app cannot collect for you, so a library
    that does not name them leaves no clue they exist. Anything already in the file
    is left alone, byte for byte; only a key that is absent gets appended.
    """
    path = root / ENV_FILE
    present = read_env_file(root)
    blocks: list[str] = []
    for description, names in _BOT_SECRETS:
        absent = [name for name in names if name not in present]
        if absent:
            blocks.append("\n".join([f"# {description}", *(f"{name}={_BLANK}" for name in absent)]))
    if not blocks:
        return path

    if path.is_file():
        preamble = path.read_text(encoding="utf-8").rstrip("\n")
    else:
        preamble = f"{_ENV_HEADER}\n\n# Fill in a pair to run that bot; the README walks through both."
    parts = [part for part in (preamble, *blocks) if part]
    return _write_secrets(path, ["\n\n".join(parts), ""])


def write_env(root: Path, values: dict[str, str]) -> Path:
    """Merge ``values`` into ``.env``, dropping the ones set to empty.

    The process environment is updated too. ``load_dotenv`` refuses to overwrite
    a variable that is already set, so without this a key changed in the app
    would not take effect until the next restart.
    """
    merged = read_env_file(root)
    for name, value in values.items():
        if value:
            merged[name] = value
            os.environ[name] = value
        else:
            merged.pop(name, None)
            os.environ.pop(name, None)

    lines = [
        _ENV_HEADER,
        "",
        # A key still waiting to be filled in keeps the shape the template gave it,
        # so saving from the app does not quietly turn a reminder into a bare name.
        *(f"{name}={value or _BLANK}" for name, value in sorted(merged.items())),
        "",
    ]
    return _write_secrets(root / ENV_FILE, lines)


def _write_secrets(path: Path, lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    # Not every filesystem supports permissions; the file is still written.
    with contextlib.suppress(OSError):
        path.chmod(0o600)
    return path


# --------------------------------------------------------------- config.toml


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return _quote(str(value))


def _table(name: str, values: dict[str, Any]) -> list[str]:
    rows = [f"{key} = {_render(value)}" for key, value in values.items() if value is not None]
    return [f"[{name}]", *rows, ""] if rows else []


def _merge(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``incoming`` on ``existing``; an explicit ``None`` clears a key."""
    merged = dict(existing)
    for key, value in incoming.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict):
            merged[key] = _merge(merged.get(key) or {}, value)
        else:
            merged[key] = value
    return merged


def write_config(root: Path, payload: dict[str, Any]) -> Path:
    """Write ``config.toml``, keeping settings the app does not know about.

    The app only edits the model, language and arXiv sections. Bot credentials
    and anything else added by hand are read back and written out unchanged, so
    saving from the Settings page never silently discards them.
    """
    path = root / CONFIG_FILE
    current: dict[str, Any] = {}
    if path.is_file():
        with path.open("rb") as handle:
            current = tomllib.load(handle)
    data = _merge(current, payload)

    lines = [_HEADER, f"default_language = {_quote(data.get('default_language', 'zh'))}", ""]
    for section in ("llm", "arxiv", "server", "feishu", "wecom"):
        lines += _table(section, data.get(section, {}))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def is_configured(settings: Settings | None = None) -> bool:
    """Whether the app has enough to do anything beyond downloading LaTeX."""
    settings = settings or get_settings()
    return bool(settings.llm.api_key)


def apply(root: Path, payload: dict[str, Any], secrets: dict[str, str]) -> Settings:
    """Persist a full settings payload and return the reloaded settings."""
    write_config(root, payload)
    write_env(root, secrets)
    return reload_settings()
