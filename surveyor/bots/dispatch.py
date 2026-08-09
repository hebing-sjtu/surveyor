"""What happens after a message arrives, whichever transport carried it.

A message can reach us four ways: an HTTP callback from either platform
(``surveyor serve``), or a long connection to either (``surveyor feishu-connect``,
``surveyor wecom-connect``). Past the point where one has been decoded, all of
them want the same thing: drop platform retries, acknowledge at once, then do the
slow part on a worker thread and push the result back.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Callable

from .router import BotContext, BotReply, Router

log = logging.getLogger(__name__)


class SeenCache:
    """Bounded set of handled event ids, so platform retries are not re-run."""

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, None] = OrderedDict()
        self._lock = Lock()

    def add_if_new(self, key: str) -> bool:
        if not key:
            return True
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return False
            self._items[key] = None
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
            return True


def run_conversation(
    router: Router, message: str, context: BotContext, deliver: Callable[[str], None]
) -> None:
    """Send the acknowledgement, then the slow answer. Runs on a worker thread."""
    try:
        reply: BotReply = router.handle(message, context)
    except Exception as exc:
        log.exception("router failed")
        safe_deliver(deliver, f"Sorry, that failed: {exc}")
        return

    if reply.ack:
        safe_deliver(deliver, reply.ack)
    if reply.work is None:
        return
    try:
        result = reply.work()
    except Exception as exc:
        log.exception("deferred work failed")
        result = f"Sorry, that failed: {exc}"
    if result:
        safe_deliver(deliver, result)


def safe_deliver(deliver: Callable[[str], None], text: str) -> None:
    try:
        deliver(text)
    except Exception as exc:
        log.error("could not deliver message: %s", exc)
