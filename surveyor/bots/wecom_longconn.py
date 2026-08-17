"""WeCom events over a long connection, instead of an HTTP callback.

This is a different bot from the self-built app in ``wecom.py``, not a second mode
of it. WeCom offers the long connection to 智能机器人 only: it authenticates with a
BotID and a long-connection Secret, and carries JSON frames rather than encrypted
XML, so nothing about the app's token, AES key or agent id applies here. The two
API modes are mutually exclusive on the console, so a bot is either reachable at a
callback URL or over this socket, never both.

What it buys is the same thing ``feishu_longconn`` buys: the machine dials out, so
a library on a laptop needs no public URL and no tunnel.

Answers take minutes, which shapes how replies go out. The acknowledgement travels
back down the callback's own reply channel, and the answer that follows is pushed
to the conversation later — the mechanism WeCom documents for exactly this, and one
that avoids the ten-minute ceiling on a streamed reply.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..config import Settings, get_settings
from ..store import PaperStore
from .dispatch import SeenCache, run_conversation
from .router import BotContext, Router, split_message
from .wecom import MESSAGE_LIMIT

log = logging.getLogger(__name__)

SDK_MISSING = (
    "A long connection needs the official WeCom smart-robot SDK, which is not "
    "installed.\nInstall it with:  pip install wecom-aibot-python-sdk"
)

# A reply frame is acknowledged within seconds or not at all; this only has to be
# long enough to cover a slow round trip.
DELIVER_TIMEOUT = 30.0

# How a push says which kind of conversation it is aiming at. Worth being explicit:
# left out, WeCom guesses, and it guesses group.
SINGLE_CHAT = 1
GROUP_CHAT = 2


def _load_sdk() -> Any:
    try:
        import aibot
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(SDK_MISSING) from exc
    return aibot


def extract_message(body: dict[str, Any]) -> tuple[str, BotContext] | None:
    """Pull ``(text, context)`` out of a long-connection message callback.

    The frame is JSON with its own field names, so this cannot be shared with the
    XML callback path in ``wecom.py`` — only the ``BotContext`` it produces is.
    """
    if body.get("msgtype") != "text":
        return None
    text = (body.get("text") or {}).get("content", "")
    if not text.strip():
        return None

    is_group = body.get("chattype") == "group"
    user_id = (body.get("from") or {}).get("userid", "unknown")
    context = BotContext(
        platform="wecom",
        user_id=user_id,
        # A push needs an id for the conversation, and for a single chat that is
        # the person, not a chat that has one.
        chat_id=body.get("chatid", "") if is_group else user_id,
        is_group=is_group,
    )
    return text, context


def run(settings: Settings | None = None) -> None:
    """Receive and answer WeCom messages until interrupted.

    Blocks. The SDK keeps the socket alive and reconnects on its own, so this
    returns only on Ctrl+C or an authentication failure.
    """
    # Configuration is checked before the import, so that someone who has not
    # turned the bot on hears about that rather than about a missing package.
    settings = settings or get_settings()
    if not settings.wecom.enabled:
        raise RuntimeError(
            "The WeCom bot is switched off. Turn it on under Settings in the app, "
            f"or set `enabled = true` under [wecom] in {settings.root / 'config.toml'}."
        )
    if not (settings.wecom.bot_id and settings.wecom.bot_secret):
        raise RuntimeError(
            f"{settings.wecom.bot_id_env} and {settings.wecom.bot_secret_env} are not "
            f"set. Put them in {settings.root / '.env'} and try again. They come from "
            "the smart robot's own page, and are not the app's token or AES key."
        )

    aibot = _load_sdk()
    router = Router(PaperStore(settings))
    workers = ThreadPoolExecutor(
        max_workers=settings.server.max_workers, thread_name_prefix="surveyor"
    )
    try:
        asyncio.run(_serve(aibot, settings, router, workers))
    finally:
        workers.shutdown(wait=False, cancel_futures=True)


async def _serve(
    aibot: Any, settings: Settings, router: Router, workers: ThreadPoolExecutor
) -> None:
    """Hold the connection open, handing each message to a worker thread."""
    loop = asyncio.get_running_loop()
    seen = SeenCache()
    client = aibot.WSClient(
        aibot.WSClientOptions(
            bot_id=settings.wecom.bot_id,
            secret=settings.wecom.bot_secret,
            ws_url=settings.wecom.ws_url,
            # A laptop's network comes and goes; giving up after ten tries would
            # leave the bot silently dead after a long sleep.
            max_reconnect_attempts=-1,
        )
    )

    def on_text(frame: dict[str, Any]) -> None:
        """Runs on the event loop, so it hands off rather than doing any work."""
        body = frame.get("body") or {}
        extracted = extract_message(body)
        if extracted is None:
            return
        if not seen.add_if_new(body.get("msgid", "")):
            log.info("ignoring duplicate wecom message")
            return

        message, context = extracted
        log.info("%s asked: %s", context.user_id, message[:80])
        deliver = _deliver_to(client, loop, frame, context)
        workers.submit(run_conversation, router, message, context, deliver)

    client.on("message.text", on_text)
    client.on("error", lambda exc: log.error("wecom connection: %s", exc))
    await client.connect()
    try:
        await asyncio.Event().wait()
    finally:
        client.disconnect()


def _deliver_to(
    client: Any,
    loop: asyncio.AbstractEventLoop,
    frame: dict[str, Any],
    context: BotContext,
) -> Any:
    """Build the callback that sends text back for one incoming message.

    Called from a worker thread while the socket lives on the event loop, so every
    send is handed to that loop and waited on. The first call answers the callback
    directly; later ones push, since by then WeCom considers the reply channel's
    job done.
    """
    stream_id = uuid.uuid4().hex
    answered = False

    # Unlike every other route, a smart robot can only send markdown or a
    # template card, so replies here cannot be drawn as pictures.
    def push(piece: str) -> None:
        body = {
            "chat_type": GROUP_CHAT if context.is_group else SINGLE_CHAT,
            "msgtype": "markdown",
            "markdown": {"content": piece},
        }
        try:
            _run(loop, client.send_message(context.chat_id, body))
        except Exception as exc:
            # A push can be refused where a reply is still accepted, and an answer
            # nobody sees is the one failure worth a second attempt.
            log.warning("push failed (%s); replying on the callback instead", exc)
            _run(loop, client.reply_stream(frame, uuid.uuid4().hex, piece, finish=True))

    def deliver(text: str) -> None:
        nonlocal answered
        # A listing can be longer than WeCom accepts in one message, so this splits
        # first and only then decides which channel each piece travels on.
        for piece in split_message(text, MESSAGE_LIMIT):
            if answered:
                push(piece)
            else:
                answered = True
                _run(loop, client.reply_stream(frame, stream_id, piece, finish=True))

    return deliver


def _run(loop: asyncio.AbstractEventLoop, coro: Any) -> Any:
    return asyncio.run_coroutine_threadsafe(coro, loop).result(DELIVER_TIMEOUT)
