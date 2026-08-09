"""Configuration loading.

Settings come from three layers, later ones winning: built-in defaults,
``config.toml`` in the library root, then environment variables (``.env`` is
loaded automatically). Secrets belong in the environment, never in config.toml.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv
from pydantic import BaseModel, Field

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

LIBRARY_MARKERS = ("config.toml", "papers", "knowledge")


def app_config_dir() -> Path:
    """Per-user settings that live outside any one library."""
    if os.name == "nt":
        base = os.getenv("APPDATA") or Path.home() / "AppData" / "Roaming"
    else:
        base = os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "surveyor"


def default_library_root() -> Path:
    return Path.home() / "Surveyor"


def saved_library_root() -> Path | None:
    """The library the user last chose in the app, if any."""
    pointer = app_config_dir() / "home"
    if not pointer.is_file():
        return None
    try:
        value = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return Path(value).expanduser() if value else None


def looks_like_library(path: Path) -> bool:
    return any((path / name).exists() for name in LIBRARY_MARKERS)


def library_root() -> Path:
    """Where the paper library lives.

    ``SURVEYOR_HOME`` wins, so a single shell can be pointed anywhere. Then the
    folder chosen in the app, which is what makes an installed copy remember
    where its data is. Failing both, a directory that already looks like a
    library is used when you are standing in one, and otherwise a fresh library
    is created under the home directory.
    """
    configured = os.getenv("SURVEYOR_HOME")
    if configured:
        return Path(configured).expanduser().resolve()

    saved = saved_library_root()
    if saved:
        return saved.resolve()

    cwd = Path.cwd().resolve()
    if looks_like_library(cwd):
        return cwd
    return default_library_root().resolve()


class Language(str, Enum):
    ZH = "zh"
    EN = "en"
    BILINGUAL = "bilingual"

    @property
    def instruction(self) -> str:
        """The wording injected into prompts to pin the output language."""
        if self is Language.ZH:
            return (
                "Write your entire response in Simplified Chinese. Keep technical terms, "
                "model names, dataset names and metric names in their original English, "
                "optionally with a Chinese gloss on first use, e.g. 对比学习 (contrastive learning). "
                "Never translate LaTeX math or code."
            )
        if self is Language.EN:
            return "Write your entire response in English."
        return (
            "Write every section twice: first in Simplified Chinese, then the same content "
            "in English under an '---' separator. Keep technical terms in English in both."
        )

    @classmethod
    def parse(cls, value: str | None, default: "Language") -> "Language":
        if not value:
            return default
        normalized = value.strip().lower()
        aliases = {
            "chinese": cls.ZH,
            "zh-cn": cls.ZH,
            "zh_cn": cls.ZH,
            "中文": cls.ZH,
            "english": cls.EN,
            "en-us": cls.EN,
            "英文": cls.EN,
            "both": cls.BILINGUAL,
            "bi": cls.BILINGUAL,
            "双语": cls.BILINGUAL,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return default


class LLMConfig(BaseModel):
    """Any OpenAI-compatible chat-completions endpoint."""

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    # Used for the high-volume per-chunk passes; falls back to `model`.
    fast_model: str | None = None
    api_key_env: str = "SURVEYOR_API_KEY"
    temperature: float = 0.2
    timeout: float = 300.0
    max_retries: int = 4
    # Characters, not tokens: a provider-agnostic budget for one request's context.
    # Roughly 3.5 chars/token for English prose, so 400k chars ~ 115k tokens.
    max_context_chars: int = 400_000

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) or None

    def model_for(self, tier: str) -> str:
        if tier == "fast" and self.fast_model:
            return self.fast_model
        return self.model


class ArxivConfig(BaseModel):
    # arXiv asks automated clients to identify themselves and to stay under
    # one request per 3 seconds.
    user_agent: str = "surveyor/0.1 (https://github.com/; mailto:surveyor@example.com)"
    request_interval: float = 3.0
    timeout: float = 120.0


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    # Bounds how many bot questions are answered concurrently.
    max_workers: int = 4


class FeishuConfig(BaseModel):
    enabled: bool = False
    app_id_env: str = "FEISHU_APP_ID"
    app_secret_env: str = "FEISHU_APP_SECRET"
    # Optional: only set these if you enabled encryption / token verification.
    encrypt_key_env: str = "FEISHU_ENCRYPT_KEY"
    verification_token_env: str = "FEISHU_VERIFICATION_TOKEN"
    # open.feishu.cn for the China site, open.larksuite.com for Lark international.
    base_url: str = "https://open.feishu.cn/open-apis"

    @property
    def app_id(self) -> str | None:
        return os.getenv(self.app_id_env) or None

    @property
    def app_secret(self) -> str | None:
        return os.getenv(self.app_secret_env) or None

    @property
    def encrypt_key(self) -> str | None:
        return os.getenv(self.encrypt_key_env) or None

    @property
    def verification_token(self) -> str | None:
        return os.getenv(self.verification_token_env) or None


class WecomConfig(BaseModel):
    enabled: bool = False
    token_env: str = "WECOM_TOKEN"
    aes_key_env: str = "WECOM_AES_KEY"
    # For a self-built app this is the CorpID; for a smart robot it is the ReceiveID
    # shown on the callback settings page.
    receive_id_env: str = "WECOM_RECEIVE_ID"
    corp_id_env: str = "WECOM_CORP_ID"
    corp_secret_env: str = "WECOM_CORP_SECRET"
    agent_id_env: str = "WECOM_AGENT_ID"
    # Group-robot incoming webhook, used for pushing digests.
    webhook_url_env: str = "WECOM_WEBHOOK_URL"
    # A 智能机器人 on a long connection is a different animal from the self-built app
    # above: it authenticates with its own pair of credentials and speaks its own
    # frame protocol, so none of the values above apply to it.
    bot_id_env: str = "WECOM_BOT_ID"
    bot_secret_env: str = "WECOM_BOT_SECRET"
    base_url: str = "https://qyapi.weixin.qq.com/cgi-bin"
    # Private deployments publish their own gateway on the admin console.
    ws_url: str = "wss://openws.work.weixin.qq.com"

    @property
    def token(self) -> str | None:
        return os.getenv(self.token_env) or None

    @property
    def aes_key(self) -> str | None:
        return os.getenv(self.aes_key_env) or None

    @property
    def receive_id(self) -> str:
        return os.getenv(self.receive_id_env) or os.getenv(self.corp_id_env) or ""

    @property
    def corp_id(self) -> str | None:
        return os.getenv(self.corp_id_env) or None

    @property
    def corp_secret(self) -> str | None:
        return os.getenv(self.corp_secret_env) or None

    @property
    def agent_id(self) -> str | None:
        return os.getenv(self.agent_id_env) or None

    @property
    def webhook_url(self) -> str | None:
        return os.getenv(self.webhook_url_env) or None

    @property
    def bot_id(self) -> str | None:
        return os.getenv(self.bot_id_env) or None

    @property
    def bot_secret(self) -> str | None:
        return os.getenv(self.bot_secret_env) or None


class Settings(BaseModel):
    root: Path = Field(default_factory=library_root)
    papers_dirname: str = "papers"
    knowledge_dirname: str = "knowledge"

    default_language: Language = Language.ZH
    llm: LLMConfig = Field(default_factory=LLMConfig)
    arxiv: ArxivConfig = Field(default_factory=ArxivConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    wecom: WecomConfig = Field(default_factory=WecomConfig)

    @property
    def papers_dir(self) -> Path:
        return self.root / self.papers_dirname

    @property
    def knowledge_dir(self) -> Path:
        return self.root / self.knowledge_dirname

    @property
    def state_dir(self) -> Path:
        return self.root / ".surveyor"

    def ensure_dirs(self) -> None:
        for path in (self.papers_dir, self.knowledge_dir, self.state_dir):
            path.mkdir(parents=True, exist_ok=True)
        # Imported here rather than at the top: writing settings is the other
        # module's job, and it reads this one.
        from .userconfig import ensure_env_file

        ensure_env_file(self.root)


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Environment overrides for the handful of knobs worth changing per-shell."""
    env_map = {
        "SURVEYOR_BASE_URL": ("llm", "base_url"),
        "SURVEYOR_MODEL": ("llm", "model"),
        "SURVEYOR_FAST_MODEL": ("llm", "fast_model"),
        "SURVEYOR_LANGUAGE": ("default_language",),
        "SURVEYOR_PORT": ("server", "port"),
    }
    for env_name, keys in env_map.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        cursor = data
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = int(raw) if keys[-1] == "port" else raw
    return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = library_root()
    load_dotenv(root / ".env")
    if root != PACKAGE_ROOT:
        # A source checkout may hold the keys even when the library lives elsewhere.
        load_dotenv(PACKAGE_ROOT / ".env")
    data = _apply_env_overrides(_read_config_file(root / "config.toml"))
    data.setdefault("root", root)
    return Settings(**data)


def reload_settings() -> Settings:
    """Re-read configuration after the app has written it."""
    get_settings.cache_clear()
    return get_settings()
