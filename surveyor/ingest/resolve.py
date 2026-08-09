"""Turn whatever the user pasted into a list of arXiv identifiers.

Accepted forms, mixed freely in one call:

* bare identifiers, ``2503.07598`` / ``2503.07598v2`` / ``arXiv:2503.07598``
* legacy identifiers, ``hep-th/9901001`` / ``math.GT/0309136``
* any URL containing one, including ``/abs/``, ``/pdf/``, ``/html/``, ``/e-print/``
  on arxiv.org as well as mirrors like huggingface.co/papers and alphaxiv.org
* a path to a ``.txt`` / ``.md`` / ``.csv`` file listing any of the above, one per
  line or embedded in Markdown bullets and links
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

# 0704.0001 onwards. Requires a 4-digit YYMM so ordinary decimals do not match.
_NEW_STYLE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")
# Pre-2007 identifiers such as hep-th/9901001 or math.GT/0309136.
_OLD_STYLE = re.compile(r"\b([a-z][a-z-]{2,}(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")

_LIST_FILE_SUFFIXES = {".txt", ".md", ".csv", ".list", ".tsv"}


@dataclass(frozen=True)
class ArxivRef:
    """A parsed reference. ``version`` is None when the user did not pin one."""

    base_id: str
    version: str | None = None

    @property
    def canonical(self) -> str:
        """The id to request from arXiv, version included when known."""
        return f"{self.base_id}{self.version or ''}"

    @property
    def dirname(self) -> str:
        """Filesystem-safe directory name; legacy ids contain a slash."""
        return self.canonical.replace("/", "_")

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.canonical


def parse_ref(text: str) -> ArxivRef | None:
    """Extract the first arXiv reference from a string, if any."""
    refs = extract_refs(text)
    return refs[0] if refs else None


def extract_refs(text: str) -> list[ArxivRef]:
    """Extract every arXiv reference from a blob of text, preserving order."""
    found: list[ArxivRef] = []
    seen: set[str] = set()
    for pattern in (_NEW_STYLE, _OLD_STYLE):
        for match in pattern.finditer(text):
            base, version = match.group(1), match.group(2)
            # A trailing ".pdf" is part of the URL, not the identifier.
            if base.endswith(".pdf"):
                base = base[:-4]
            key = f"{base}{version or ''}"
            if key in seen:
                continue
            seen.add(key)
            found.append(ArxivRef(base_id=base, version=version))
    # Sort by position in the original text so a mixed-style list keeps its order.
    positions = {ref.canonical: text.find(ref.base_id) for ref in found}
    return sorted(found, key=lambda ref: positions[ref.canonical])


def _looks_like_list_file(item: str) -> bool:
    if "\n" in item or len(item) > 512:
        return False
    path = Path(item).expanduser()
    return path.suffix.lower() in _LIST_FILE_SUFFIXES and path.is_file()


def resolve_inputs(items: Sequence[str]) -> list[ArxivRef]:
    """Resolve CLI arguments / message text into a deduplicated list of refs.

    A later reference that pins a version supersedes an earlier unpinned one for
    the same paper, so ``paper add 2503.07598 2503.07598v2`` ingests one paper.
    """
    blobs: list[str] = []
    for item in items:
        if _looks_like_list_file(item):
            blobs.append(Path(item).expanduser().read_text(encoding="utf-8", errors="replace"))
        else:
            blobs.append(item)
    return dedupe_refs(ref for blob in blobs for ref in extract_refs(blob))


def dedupe_refs(refs: Iterable[ArxivRef]) -> list[ArxivRef]:
    """Collapse refs to one per base id, preferring an explicit version."""
    ordered: dict[str, ArxivRef] = {}
    for ref in refs:
        existing = ordered.get(ref.base_id)
        if existing is None or (existing.version is None and ref.version is not None):
            ordered[ref.base_id] = ref
    return list(ordered.values())
