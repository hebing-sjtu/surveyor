"""WeCom (企业微信) adapter: callback decoding and outbound messages.

Targets a self-built app, which is the flow that supports both receiving
messages and pushing replies later. Because WeCom expects the callback response
within five seconds and our answers take minutes, callbacks are acknowledged
empty and the real reply is pushed through the message API.
"""

from __future__ import annotations

import html
import logging
import threading
import time
from typing import Any
from xml.etree import ElementTree

import httpx

from ..config import WecomConfig
from .crypto import CryptoError, wecom_decrypt, wecom_signature
from .router import BotContext, split_message

log = logging.getLogger(__name__)

# WeCom caps markdown bodies at 4096 bytes; stay clear of the edge.
MESSAGE_LIMIT = 3800


class WecomError(RuntimeError):
    pass


class WecomClient:
    def __init__(self, config: WecomConfig) -> None:
        self.config = config
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.corp_id and self.config.corp_secret and self.config.agent_id)

    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at:
                return self._token
            if not self.is_configured:
                raise WecomError(
                    "WECOM_CORP_ID / WECOM_CORP_SECRET / WECOM_AGENT_ID are not set"
                )
            response = httpx.get(
                f"{self.config.base_url}/gettoken",
                params={"corpid": self.config.corp_id, "corpsecret": self.config.corp_secret},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("errcode") != 0:
                raise WecomError(f"token request failed: {data}")
            self._token = data["access_token"]
            self._expires_at = time.time() + max(int(data.get("expires_in", 7200)) - 60, 60)
            return self._token

    def send(self, user_id: str, text: str, *, chat_id: str = "") -> None:
        """Push a markdown reply to a user, or to an app chat when given one."""
        for piece in split_message(text, MESSAGE_LIMIT):
            payload: dict[str, Any] = {
                "msgtype": "markdown",
                "agentid": int(self.config.agent_id or 0),
                "markdown": {"content": piece},
            }
            if chat_id:
                payload["chatid"] = chat_id
                endpoint = "appchat/send"
            else:
                payload["touser"] = user_id
                endpoint = "message/send"

            response = httpx.post(
                f"{self.config.base_url}/{endpoint}",
                params={"access_token": self.token()},
                json=payload,
                timeout=30,
            )
            data = response.json()
            if data.get("errcode") != 0:
                log.error("wecom send failed: %s", data)
                raise WecomError(f"send failed: {data}")

    def push_webhook(self, text: str) -> None:
        """Post to a group robot's incoming webhook, used for scheduled digests."""
        url = self.config.webhook_url
        if not url:
            raise WecomError("WECOM_WEBHOOK_URL is not set")
        for piece in split_message(text, MESSAGE_LIMIT):
            response = httpx.post(
                url, json={"msgtype": "markdown", "markdown": {"content": piece}}, timeout=30
            )
            data = response.json()
            if data.get("errcode") != 0:
                raise WecomError(f"webhook push failed: {data}")


def verify_url(config: WecomConfig, params: dict[str, str]) -> str:
    """Handle the GET WeCom sends when you save the callback URL."""
    token, aes_key = config.token, config.aes_key
    if not token or not aes_key:
        raise CryptoError("WECOM_TOKEN / WECOM_AES_KEY are not set")

    echostr = params.get("echostr", "")
    expected = wecom_signature(
        token, params.get("timestamp", ""), params.get("nonce", ""), echostr
    )
    if expected != params.get("msg_signature", ""):
        raise CryptoError("msg_signature mismatch")

    message, receive_id = wecom_decrypt(aes_key, echostr)
    _check_receive_id(config, receive_id)
    return message


def decode_request(config: WecomConfig, raw_body: bytes, params: dict[str, str]) -> dict[str, str]:
    """Verify and decrypt a callback POST into a flat dict of XML fields."""
    token, aes_key = config.token, config.aes_key
    if not token or not aes_key:
        raise CryptoError("WECOM_TOKEN / WECOM_AES_KEY are not set")

    envelope = _parse_xml(raw_body.decode("utf-8", errors="replace"))
    encrypted = envelope.get("Encrypt")
    if not encrypted:
        raise CryptoError("callback body has no <Encrypt> element")

    expected = wecom_signature(
        token, params.get("timestamp", ""), params.get("nonce", ""), encrypted
    )
    if expected != params.get("msg_signature", ""):
        raise CryptoError("msg_signature mismatch")

    plaintext, receive_id = wecom_decrypt(aes_key, encrypted)
    _check_receive_id(config, receive_id)
    return _parse_xml(plaintext)


def _check_receive_id(config: WecomConfig, receive_id: str) -> None:
    expected = config.receive_id
    if expected and receive_id and receive_id != expected:
        raise CryptoError(f"unexpected receive id {receive_id!r}")


def _parse_xml(text: str) -> dict[str, str]:
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise CryptoError(f"malformed XML: {exc}") from exc
    return {child.tag: (child.text or "") for child in root}


def extract_message(fields: dict[str, str]) -> tuple[str, BotContext, str] | None:
    """Pull ``(text, context, message_id)`` out of a decoded callback."""
    if fields.get("MsgType") != "text":
        return None
    text = html.unescape(fields.get("Content", ""))
    if not text.strip():
        return None

    chat_id = fields.get("ChatId", "")
    context = BotContext(
        platform="wecom",
        user_id=fields.get("FromUserName", "unknown"),
        chat_id=chat_id,
        is_group=bool(chat_id),
    )
    return text, context, fields.get("MsgId", "")
