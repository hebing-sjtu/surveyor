"""Background work for the app.

Downloading a survey and summarizing it takes minutes, far longer than a browser
request should wait. Anything slow is therefore started as a job: the request
returns an id immediately and the page polls for the log as it fills in.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

# Enough history for a working session; old jobs are only useful as a log.
MAX_JOBS = 60


@dataclass
class Job:
    id: str
    label: str
    status: str = "running"  # running | done | failed | cancelled
    lines: list[str] = field(default_factory=list)
    result: Any = None
    error: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def as_dict(self, since: int = 0) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "lines": self.lines[since:],
            "total_lines": len(self.lines),
            "result": self.result,
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
        }


class JobRunner:
    """A small thread pool with a log per job.

    Serialized on purpose: arXiv asks for one request every three seconds, and
    running two ingestions at once would race on the same library folder.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._lock = threading.Lock()
        self._gate = threading.Semaphore(1)

    def start(self, label: str, work: Callable[[Callable[[str], None]], Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], label=label)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > MAX_JOBS:
                self._jobs.pop(self._order.popleft(), None)

        def progress(message: str) -> None:
            with self._lock:
                job.lines.append(message)
            log.info("[%s] %s", job.label, message)

        def run() -> None:
            queued = not self._gate.acquire(blocking=False)
            if queued:
                progress("waiting for the current task to finish")
                self._gate.acquire()
            try:
                job.result = work(progress)
                job.status = "done"
            except Exception as exc:  # surfaced in the UI, never crashes the server
                job.status = "failed"
                job.error = str(exc)
                progress(f"failed: {exc}")
                log.exception("job %s failed", job.label)
            finally:
                job.finished_at = time.time()
                self._gate.release()

        threading.Thread(target=run, name=f"job-{job.id}", daemon=True).start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            ids = list(self._order)[-limit:]
            return [self._jobs[job_id] for job_id in reversed(ids) if job_id in self._jobs]

    @property
    def busy(self) -> bool:
        with self._lock:
            return any(job.status == "running" for job in self._jobs.values())
