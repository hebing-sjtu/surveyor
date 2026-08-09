"""Markdown to HTML for the browser views.

Raw HTML in the source is escaped rather than passed through: notes are written
by a language model from text found in arXiv uploads, so neither end of that
chain is trusted markup.
"""

from __future__ import annotations

from functools import lru_cache

from markdown_it import MarkdownIt


@lru_cache(maxsize=1)
def _parser() -> MarkdownIt:
    return (
        MarkdownIt("commonmark", {"html": False, "linkify": False})
        .enable("table")
        .enable("strikethrough")
    )


def to_html(markdown: str) -> str:
    return _parser().render(markdown or "")
