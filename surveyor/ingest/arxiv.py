"""arXiv metadata lookup and TeX source ("e-print") download."""

from __future__ import annotations

import gzip
import io
import logging
import re
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import httpx

from ..config import ArxivConfig
from ..models import PaperMeta
from .resolve import ArxivRef

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
EPRINT_URL = "https://arxiv.org/e-print/{id}"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# arXiv asks for no more than one request every three seconds; the lock keeps
# that promise even if we later fan out ingestion across threads.
_rate_lock = threading.Lock()
_last_request_at = 0.0


class ArxivError(RuntimeError):
    pass


def _throttle(interval: float) -> None:
    global _last_request_at
    with _rate_lock:
        wait = interval - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


def _client(config: ArxivConfig) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": config.user_agent},
        timeout=config.timeout,
        follow_redirects=True,
    )


def _text(node: ElementTree.Element | None) -> str:
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def _parse_entry(entry: ElementTree.Element) -> PaperMeta | None:
    raw_id = _text(entry.find(f"{_ATOM}id"))
    if not raw_id or "api/errors" in raw_id:
        return None

    # The id looks like http://arxiv.org/abs/2503.07598v2
    tail = raw_id.rsplit("/abs/", 1)[-1]
    base_id, _, version = tail.partition("v")
    version = f"v{version}" if version else None

    categories = [
        node.get("term", "")
        for node in entry.findall(f"{_ATOM}category")
        if node.get("term")
    ]
    primary = entry.find(f"{_ARXIV}primary_category")

    return PaperMeta(
        paper_id=tail.replace("/", "_"),
        arxiv_id=base_id,
        version=version,
        title=_text(entry.find(f"{_ATOM}title")),
        authors=[
            _text(node.find(f"{_ATOM}name"))
            for node in entry.findall(f"{_ATOM}author")
            if _text(node.find(f"{_ATOM}name"))
        ],
        abstract=_text(entry.find(f"{_ATOM}summary")),
        categories=categories,
        primary_category=primary.get("term") if primary is not None else None,
        published=_text(entry.find(f"{_ATOM}published")) or None,
        updated=_text(entry.find(f"{_ATOM}updated")) or None,
        doi=_text(entry.find(f"{_ARXIV}doi")) or None,
        journal_ref=_text(entry.find(f"{_ARXIV}journal_ref")) or None,
        comment=_text(entry.find(f"{_ARXIV}comment")) or None,
        abs_url=f"https://arxiv.org/abs/{tail}",
        pdf_url=f"https://arxiv.org/pdf/{tail}",
        source_kind="arxiv",
    )


def fetch_metadata(refs: list[ArxivRef], config: ArxivConfig) -> dict[str, PaperMeta]:
    """Look up metadata for several papers, keyed by the requested base id.

    Papers arXiv does not know about are simply absent from the result.
    """
    if not refs:
        return {}

    results: dict[str, PaperMeta] = {}
    batch_size = 50
    with _client(config) as client:
        for start in range(0, len(refs), batch_size):
            batch = refs[start : start + batch_size]
            _throttle(config.request_interval)
            response = client.get(
                API_URL,
                params={
                    "id_list": ",".join(ref.canonical for ref in batch),
                    "max_results": len(batch),
                },
            )
            response.raise_for_status()
            try:
                root = ElementTree.fromstring(response.text)
            except ElementTree.ParseError as exc:
                raise ArxivError(f"arXiv returned malformed XML: {exc}") from exc

            for entry in root.findall(f"{_ATOM}entry"):
                meta = _parse_entry(entry)
                if meta and meta.arxiv_id:
                    results[meta.arxiv_id] = meta
    return results


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _titles_match(wanted: str, found: str) -> bool:
    """Accept a title hit only on a near-exact match.

    Bibliographies are noisy, and a loose match here would silently attach the
    wrong arXiv id to a reference, which is worse than leaving it unresolved.
    """
    left, right = _normalize_title(wanted), _normalize_title(found)
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True
    left_words, right_words = set(left.split()), set(right.split())
    overlap = len(left_words & right_words)
    return overlap / max(len(left_words | right_words), 1) >= 0.85


def search_by_title(title: str, config: ArxivConfig) -> PaperMeta | None:
    """Find a paper on arXiv by title, for references that cite a venue version."""
    cleaned = _normalize_title(title)
    if len(cleaned) < 12:
        return None

    with _client(config) as client:
        _throttle(config.request_interval)
        try:
            response = client.get(
                API_URL,
                params={
                    "search_query": f'ti:"{cleaned}"',
                    "max_results": 5,
                },
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.text)
        except (httpx.HTTPError, ElementTree.ParseError) as exc:
            log.debug("title search failed for %r: %s", title[:60], exc)
            return None

    for entry in root.findall(f"{_ATOM}entry"):
        meta = _parse_entry(entry)
        if meta and _titles_match(title, meta.title):
            return meta
    return None


def download_source(ref: ArxivRef, config: ArxivConfig) -> bytes:
    """Fetch the raw e-print archive bytes for one paper."""
    url = EPRINT_URL.format(id=ref.canonical)
    with _client(config) as client:
        _throttle(config.request_interval)
        response = client.get(url)
    if response.status_code == 404:
        raise ArxivError(f"{ref.canonical}: no e-print available (withdrawn or PDF-only)")
    response.raise_for_status()
    if not response.content:
        raise ArxivError(f"{ref.canonical}: arXiv returned an empty e-print")
    return response.content


def _is_safe_member(name: str) -> bool:
    """Reject absolute paths and ``..`` traversal from untrusted archives."""
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def extract_source(data: bytes, dest: Path) -> str:
    """Unpack an e-print payload into ``dest``. Returns the detected format.

    arXiv hands back a gzipped tar for most submissions, a bare gzipped file for
    single-file ones, and occasionally a PDF when no source was submitted.
    """
    dest.mkdir(parents=True, exist_ok=True)

    if data[:4] == b"%PDF":
        (dest / "paper.pdf").write_bytes(data)
        return "pdf"

    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                if _is_safe_member(member):
                    archive.extract(member, dest)
        return "zip"

    payload = data
    if data[:2] == b"\x1f\x8b":
        payload = gzip.decompress(data)
        if payload[:4] == b"%PDF":
            (dest / "paper.pdf").write_bytes(payload)
            return "pdf"

    try:
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            members = [m for m in archive.getmembers() if _is_safe_member(m.name)]
            archive.extractall(dest, members=members)
        return "tar"
    except tarfile.TarError:
        pass

    # Single-file submission: a lone .tex, gzipped.
    (dest / "main.tex").write_bytes(payload)
    return "tex"


def fetch_and_extract(ref: ArxivRef, dest: Path, config: ArxivConfig) -> str:
    return extract_source(download_source(ref, config), dest)
