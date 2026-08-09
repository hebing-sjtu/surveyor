"""Feishu events over a long connection, instead of an HTTP callback.

The webhook route needs a public HTTPS URL, a request-URL handshake and — if you
turn encryption on — a shared key. A long connection needs none of that: the
official SDK dials out to Feishu over a WebSocket and authenticates with the app
credentials, so events arrive on a socket this machine opened itself. That makes
it the practical choice for a library that lives on a laptop.

Only the transport differs. Decoding an event and answering it stay shared with
``server.py``, so a question asked over a long connection behaves identically.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import Settings, get_settings
from ..store import PaperStore
from .dispatch import SeenCache, run_conversation
from .feishu import FeishuClient, event_id, extract_message
from .router import Router

log = logging.getLogger(__name__)

SDK_MISSING = (
    "A long connection needs the official Feishu SDK, which is not installed.\n"
    "Install it with:  pip install lark-oapi"
)


def _load_sdk() -> Any:
    try:
        import lark_oapi
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(SDK_MISSING) from exc
    return lark_oapi


def ws_domain(base_url: str) -> str:
    """The SDK wants the bare host; ``base_url`` carries the REST prefix too.

    Deriving one from the other keeps Lark international working off the single
    ``base_url`` setting, with no second knob to forget.
    """
    return base_url.split("/open-apis")[0].rstrip("/")


def run(settings: Settings | None = None) -> None:
    """Receive and answer Feishu messages until interrupted.

    Blocks. The SDK reconnects on its own if the socket drops, so this returns
    only on Ctrl+C or an authentication failure.
    """
    # Configuration is checked before the import, so that someone who has not
    # turned the bot on hears about that rather than about a missing package.
    settings = settings or get_settings()
    if not settings.feishu.enabled:
        raise RuntimeError(
            "The Feishu bot is switched off. Turn it on under Settings in the app, "
            f"or set `enabled = true` under [feishu] in {settings.root / 'config.toml'}."
        )

    feishu = FeishuClient(settings.feishu)
    if not feishu.is_configured:
        raise RuntimeError(
            f"{settings.feishu.app_id_env} and {settings.feishu.app_secret_env} are not "
            f"set. Put them in {settings.root / '.env'} and try again."
        )

    lark = _load_sdk()
    router = Router(PaperStore(settings))
    seen = SeenCache()
    workers = ThreadPoolExecutor(
        max_workers=settings.server.max_workers, thread_name_prefix="surveyor"
    )

    def on_message(event: Any) -> None:
        """Hand the message off. Runs on the SDK's event loop, so it never blocks.

        The event arrives as SDK objects; turning it back into the callback's
        JSON shape lets both transports share one extraction path.
        """
        body = json.loads(lark.JSON.marshal(event) or "{}")
        extracted = extract_message(body)
        if extracted is None:
            return
        if not seen.add_if_new(event_id(body)):
            log.info("ignoring duplicate feishu event")
            return

        message, context, message_id = extracted
        log.info("%s asked: %s", context.user_id, message[:80])

        def deliver(text: str) -> None:
            if message_id:
                feishu.reply(message_id, text)
            else:
                feishu.send(context.chat_id, text)

        workers.submit(run_conversation, router, message, context, deliver)

    handler = (
        lark.EventDispatcherHandler.builder("", "")
        .register_p2_im_message_receive_v1(on_message)
        .build()
    )
    client = lark.ws.Client(
        settings.feishu.app_id,
        settings.feishu.app_secret,
        event_handler=handler,
        domain=ws_domain(settings.feishu.base_url),
        log_level=lark.LogLevel.INFO,
    )
    try:
        client.start()
    finally:
        workers.shutdown(wait=False, cancel_futures=True)
