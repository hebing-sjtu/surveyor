"""HTTP front end hosting both bot webhooks.

Both platforms retry a callback that does not answer within a few seconds, and
every interesting command here takes far longer than that, so the pattern is
always: verify, acknowledge, then do the work on a background thread and push
the result back through the platform's send API.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from ..config import Settings, get_settings
from ..store import PaperStore
from . import feishu as feishu_adapter
from . import wecom as wecom_adapter
from .crypto import CryptoError
from .dispatch import SeenCache, run_conversation
from .router import BotContext, Router

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    store = PaperStore(settings)
    router = Router(store)
    seen = SeenCache()
    workers = ThreadPoolExecutor(
        max_workers=settings.server.max_workers, thread_name_prefix="surveyor"
    )

    feishu_client = feishu_adapter.FeishuClient(settings.feishu)
    wecom_client = wecom_adapter.WecomClient(settings.wecom)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        workers.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="surveyor", version="0.1.0", lifespan=lifespan)

    # ------------------------------------------------------------------ meta
    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "papers": len(store.list_ids()),
            "feishu": settings.feishu.enabled and feishu_client.is_configured,
            "wecom": settings.wecom.enabled and wecom_client.is_configured,
        }

    def switched_off(platform: str) -> Response:
        """A bot nobody turned on should say so rather than answer quietly."""
        log.warning("dropped %s callback: the bot is switched off", platform)
        return JSONResponse(
            {"error": f"The {platform} bot is switched off. Turn it on under Settings."},
            status_code=403,
        )

    # ---------------------------------------------------------------- feishu
    @app.post("/webhook/feishu")
    async def feishu_webhook(request: Request) -> Response:
        if not settings.feishu.enabled:
            return switched_off("Feishu")
        raw = await request.body()
        headers = {key.lower(): value for key, value in request.headers.items()}
        try:
            body = feishu_adapter.decode_request(settings.feishu, raw, headers)
        except CryptoError as exc:
            log.warning("rejected feishu callback: %s", exc)
            return JSONResponse({"error": str(exc)}, status_code=401)

        # Saving the callback URL in the Feishu console triggers this handshake.
        if body.get("type") == "url_verification" or "challenge" in body:
            return JSONResponse({"challenge": body.get("challenge", "")})

        extracted = feishu_adapter.extract_message(body)
        if extracted is None:
            return JSONResponse({"ok": True})

        if not seen.add_if_new(feishu_adapter.event_id(body)):
            log.info("ignoring duplicate feishu event")
            return JSONResponse({"ok": True})

        message, context, message_id = extracted

        def deliver(text: str) -> None:
            if message_id:
                feishu_client.reply(message_id, text)
            else:
                feishu_client.send(context.chat_id, text)

        workers.submit(run_conversation, router, message, context, deliver)
        return JSONResponse({"ok": True})

    # ----------------------------------------------------------------- wecom
    @app.get("/webhook/wecom")
    async def wecom_verify(request: Request) -> Response:
        if not settings.wecom.enabled:
            return switched_off("WeCom")
        params = dict(request.query_params)
        try:
            return PlainTextResponse(wecom_adapter.verify_url(settings.wecom, params))
        except CryptoError as exc:
            log.warning("rejected wecom verification: %s", exc)
            return PlainTextResponse(str(exc), status_code=401)

    @app.post("/webhook/wecom")
    async def wecom_webhook(request: Request) -> Response:
        if not settings.wecom.enabled:
            return switched_off("WeCom")
        raw = await request.body()
        params = dict(request.query_params)
        try:
            fields = wecom_adapter.decode_request(settings.wecom, raw, params)
        except CryptoError as exc:
            log.warning("rejected wecom callback: %s", exc)
            return PlainTextResponse(str(exc), status_code=401)

        extracted = wecom_adapter.extract_message(fields)
        if extracted is None:
            return PlainTextResponse("")

        if not seen.add_if_new(fields.get("MsgId", "")):
            log.info("ignoring duplicate wecom message")
            return PlainTextResponse("")

        message, context, _message_id = extracted

        def deliver(text: str) -> None:
            wecom_client.send(context.user_id, text, chat_id=context.chat_id)

        workers.submit(run_conversation, router, message, context, deliver)
        # An empty body tells WeCom "no passive reply"; the answer is pushed later.
        return PlainTextResponse("")

    # ------------------------------------------------------------------ debug
    @app.post("/webhook/debug")
    async def debug_webhook(request: Request) -> Response:
        """Exercise the router without any platform, for local testing."""
        payload = json.loads(await request.body() or b"{}")
        context = BotContext(
            platform="debug", user_id=payload.get("user_id", "local"), chat_id="local"
        )
        reply = router.handle(payload.get("text", "help"), context)
        result = reply.work() if reply.work else None
        return JSONResponse({"ack": reply.ack, "result": result})

    return app


app = None  # populated by `paper serve`; uvicorn users should call create_app()
