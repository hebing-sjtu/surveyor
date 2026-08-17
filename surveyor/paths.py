"""Where documents live inside a library, and how they link to each other.

A library can hold several sub-libraries — a collection per research area — and
each gets its own knowledge folder::

    knowledge/index.md                     the whole library
    knowledge/topics/<topic>.md
    knowledge/<collection>/index.md        one sub-library
    knowledge/<collection>/topics/<topic>.md

Because a document's depth now varies, links between them are computed rather
than written by hand.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import Settings

# Anything that is not a letter or digit in any script becomes a hyphen. Keeping
# non-ASCII letters matters: stripping them would fold every Chinese collection
# name onto the same folder, and two sub-libraries would quietly share one.
_UNSAFE = re.compile(r"[^\w]+", re.UNICODE)
# Windows refuses these as names whatever the extension.
_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{digit}" for digit in range(1, 10)),
    *(f"lpt{digit}" for digit in range(1, 10)),
}


def slugify(text: str) -> str:
    """A folder name for a collection or topic, safe on every platform."""
    slug = _UNSAFE.sub("-", (text or "").casefold()).strip("-_")
    if not slug or slug in _RESERVED:
        return f"{slug}-folder" if slug else "misc"
    return slug[:80].rstrip("-_")


def knowledge_root(settings: Settings, collection: str | None = None) -> Path:
    """The folder holding the knowledge documents for one collection."""
    name = (collection or "").strip()
    return settings.knowledge_dir / slugify(name) if name else settings.knowledge_dir


def relative_link(target: Path, from_dir: Path) -> str:
    """A Markdown-safe relative path, so links work from any nesting depth."""
    return os.path.relpath(target, from_dir).replace(os.sep, "/")


def note_link(settings: Settings, paper_id: str, note: str, from_dir: Path) -> str:
    return relative_link(settings.papers_dir / paper_id / note, from_dir)
