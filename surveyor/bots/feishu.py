"""Feishu / Lark adapter: event decoding and outbound messages."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

import httpx

from ..config import FeishuConfig
from .crypto import CryptoError, feishu_decrypt, feishu_signature
from .imaging import RenderError, markdown_to_png, should_draw
from .imaging import pages as _pages
from .router import BotContext, split_message

log = logging.getLogger(__name__)

# Feishu accepts far more, but long walls of text read badly in chat.
MESSAGE_LIMIT = 4000


class FeishuError(RuntimeError):
    pass


class FeishuClient:
    """Talks to the Feishu open API, caching the tenant access token."""

    def __init__(self, config: FeishuConfig) -> None:
        self.config = config
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return bool(self.config.app_id and self.config.app_secret)

    def token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._expires_at:
                return self._token
            if not self.is_configured:
                raise FeishuError("FEISHU_APP_ID / FEISHU_APP_SECRET are not set")

            response = httpx.post(
                f"{self.config.base_url}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.config.app_id, "app_secret": self.config.app_secret},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                raise FeishuError(f"token request failed: {data}")
            self._token = data["tenant_access_token"]
            # Renew a minute early so a request never races the expiry.
            self._expires_at = time.time() + max(int(data.get("expire", 7200)) - 60, 60)
            return self._token

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token()}",
            "Content-Type": "application/json; charset=utf-8",
        }

    def upload_image(self, png: bytes) -> str:
        """Put a PNG in Feishu's store and return the key that addresses it."""
        response = httpx.post(
            f"{self.config.base_url}/im/v1/images",
            data={"image_type": "message"},
            files={"image": ("reply.png", png, "image/png")},
            headers={"Authorization": f"Bearer {self.token()}"},
            timeout=60,
        )
        data = _json_or_error(response)
        if data.get("code") != 0:
            raise FeishuError(f"image upload failed: {data.get('msg')}")
        key = (data.get("data") or {}).get("image_key")
        if not key:
            raise FeishuError("image upload returned no image_key")
        return key

    def send(self, receive_id: str, text: str, *, receive_id_type: str = "chat_id") -> None:
        """Send a reply as a picture when it has layout, as a card otherwise."""
        for message in self._messages(text):
            response = httpx.post(
                f"{self.config.base_url}/im/v1/messages",
                params={"receive_id_type": receive_id_type},
                json={"receive_id": receive_id, **message},
                headers=self._headers(),
                timeout=30,
            )
            data = _json_or_error(response)
            if data.get("code") != 0:
                log.error("feishu send failed: %s", data)
                raise FeishuError(f"send failed: {data.get('msg')}")

    def reply(self, message_id: str, text: str) -> None:
        """Reply in-thread to a specific message."""
        for message in self._messages(text):
            response = httpx.post(
                f"{self.config.base_url}/im/v1/messages/{message_id}/reply",
                json=message,
                headers=self._headers(),
                timeout=30,
            )
            data = _json_or_error(response)
            if data.get("code") != 0:
                log.error("feishu reply failed: %s", data)
                raise FeishuError(f"reply failed: {data.get('msg')}")

    def _messages(self, text: str) -> list[dict[str, Any]]:
        """Turn one reply into the message payloads that carry it."""
        if should_draw(text, self.config.send_as_image):
            try:
                return [
                    {
                        "msg_type": "image",
                        "content": json.dumps({"image_key": self.upload_image(png)}),
                    }
                    for png in [markdown_to_png(piece) for piece in _pages(text)]
                ]
            except (RenderError, FeishuError, httpx.HTTPError) as exc:
                # A picture is nicer, but an unformatted answer beats no answer.
                log.warning("falling back to text: %s", exc)

        return [
            {
                "msg_type": "interactive",
                "content": json.dumps(_markdown_card(piece), ensure_ascii=False),
            }
            for piece in split_message(text, MESSAGE_LIMIT)
        ]


def _markdown_card(text: str) -> dict[str, Any]:
    return {
        "config": {"wide_screen_mode": True},
        "elements": [{"tag": "markdown", "content": text}],
    }


def _json_or_error(response: httpx.Response) -> dict[str, Any]:
    try:
        return response.json()
    except ValueError as exc:
        raise FeishuError(f"HTTP {response.status_code}: {response.text[:300]}") from exc


def decode_request(
    config: FeishuConfig, raw_body: bytes, headers: dict[str, str]
) -> dict[str, Any]:
    """Decrypt and validate a callback body. Raises on a failed signature check."""
    encrypt_key = config.encrypt_key
    try:
        body = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CryptoError(f"body is not JSON: {exc}") from exc

    if "encrypt" in body:
        if not encrypt_key:
            raise CryptoError("callback is encrypted but FEISHU_ENCRYPT_KEY is not set")
        signature = headers.get("x-lark-signature")
        if signature:
            expected = feishu_signature(
                headers.get("x-lark-request-timestamp", ""),
                headers.get("x-lark-request-nonce", ""),
                encrypt_key,
                raw_body,
            )
            if signature != expected:
                raise CryptoError("X-Lark-Signature mismatch")
        body = json.loads(feishu_decrypt(encrypt_key, body["encrypt"]))

    token = config.verification_token
    if token:
        supplied = body.get("token") or (body.get("header") or {}).get("token")
        if supplied and supplied != token:
            raise CryptoError("verification token mismatch")
    return body


def extract_message(body: dict[str, Any]) -> tuple[str, BotContext, str] | None:
    """Pull ``(text, context, message_id)`` out of an ``im.message.receive_v1`` event."""
    header = body.get("header") or {}
    if header.get("event_type") != "im.message.receive_v1":
        return None

    event = body.get("event") or {}
    message = event.get("message") or {}
    if message.get("message_type") != "text":
        return None

    try:
        text = json.loads(message.get("content") or "{}").get("text", "")
    except json.JSONDecodeError:
        text = ""
    if not text.strip():
        return None

    sender_id = (event.get("sender") or {}).get("sender_id") or {}
    context = BotContext(
        platform="feishu",
        user_id=sender_id.get("open_id") or sender_id.get("user_id") or "unknown",
        chat_id=message.get("chat_id", ""),
        is_group=message.get("chat_type") == "group",
    )
    return text, context, message.get("message_id", "")


def event_id(body: dict[str, Any]) -> str:
    return (body.get("header") or {}).get("event_id") or ""
