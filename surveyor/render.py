"""Markdown to HTML, for the browser views and for the pictures sent to chat.

Raw HTML in the source is escaped rather than passed through: notes are written
by a language model from text found in arXiv uploads, so neither end of that
chain is trusted markup.

Formulas are converted to MathML here rather than shipped to a JavaScript
renderer in the page. Papers are full of them, and MathML is drawn by the
browser itself, which means the same markup also renders inside the headless
browser that produces images for chat, and nothing has to be downloaded at
runtime.
"""

from __future__ import annotations

import html
import logging
import re
from functools import lru_cache

from markdown_it import MarkdownIt

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _parser() -> MarkdownIt:
    return (
        MarkdownIt("commonmark", {"html": False, "linkify": False})
        .enable("table")
        .enable("strikethrough")
    )


# Code first, so a formula inside a fenced block or a code span is left alone.
_SEGMENTS = re.compile(
    r"""
      (?P<fence>^```.*?^```)               # fenced code block
    | (?P<code>`[^`\n]+`)                  # inline code span
    | \$\$(?P<display>[^$]+?)\$\$
    | (?<![\\$])\$(?P<inline>[^$\n]+?)\$(?!\$)
    """,
    re.DOTALL | re.MULTILINE | re.VERBOSE,
)
# Private-use characters survive Markdown parsing as ordinary text.
_SENTINEL = "\ue200"
_PLACEHOLDER = re.compile(f"{_SENTINEL}(\\d+){_SENTINEL}")


# Something that will actually be drawn: an identifier, a number, an operator or
# literal text. MathML with none of these renders as nothing at all.
_HAS_CONTENT = re.compile(r"<(?:mi|mn|mo|mtext|ms)[ />]")

# The alignment scaffolding of a multi-line formula. The converter has no notion
# of it and would print the ampersands, so the lines are typeset one at a time.
_ROW_BREAK = re.compile(r"\\\\(?:\s*\[[^\]]*\])?")
_ALIGN_WRAPPER = re.compile(
    r"\\(?:begin|end)\s*\{(?:aligned|align|alignat|alignedat|gathered|gather"
    r"|split|eqnarray|cases|array|multline|flalign)\*?\}(?:\s*\{[^}]*\})?"
)
_COLUMN_BREAK = re.compile(r"(?<!\\)&")


def _rows(latex: str) -> list[str]:
    """The lines of a display formula, stripped of alignment markers."""
    bare = _ALIGN_WRAPPER.sub("", latex)
    lines = [_COLUMN_BREAK.sub(" ", row).strip() for row in _ROW_BREAK.split(bare)]
    return [row for row in lines if row]


@lru_cache(maxsize=4096)
def _convert(latex: str, display: bool) -> str | None:
    try:
        from latex2mathml.converter import convert
    except ImportError:  # pragma: no cover - optional at runtime
        return None
    try:
        mathml = convert(latex, display="block" if display else "inline")
    except Exception:
        # Real papers contain macros we never saw and the odd truncated formula.
        # Showing the source is better than showing a stack trace.
        return None
    # An unknown environment converts without complaint into empty markup, which
    # would delete the formula from the page. Show the source instead.
    if not _HAS_CONTENT.search(mathml):
        return None
    # An ampersand left in place is both a stray glyph and invalid markup.
    return None if "<mi>&</mi>" in mathml else mathml


def _to_mathml(latex: str, display: bool) -> str | None:
    """Convert one formula, or return None if it is not something we can draw."""
    if display:
        rows = _rows(latex)
        if len(rows) > 1 or _ALIGN_WRAPPER.search(latex) or _COLUMN_BREAK.search(latex):
            converted = [_convert(row, True) for row in rows]
            if not converted or any(row is None for row in converted):
                return None
            return "".join(converted)
    return _convert(latex, display)


def _park_math(markdown: str) -> tuple[str, list[tuple[str, bool]]]:
    """Swap formulas for placeholders.

    This has to happen before Markdown is parsed. Left in place, ``$a_i b_i$``
    comes back with ``<em>`` where the subscripts were, and ``\\{`` loses its
    backslash to Markdown's own escaping.
    """
    spans: list[tuple[str, bool]] = []

    def park(match: re.Match[str]) -> str:
        if match.group("fence") is not None or match.group("code") is not None:
            return match.group(0)
        display = match.group("display") is not None
        latex = (match.group("display") if display else match.group("inline")) or ""
        if not latex.strip():
            return match.group(0)
        spans.append((latex.strip(), display))
        return f"{_SENTINEL}{len(spans) - 1}{_SENTINEL}"

    return _SEGMENTS.sub(park, markdown), spans


def _unpark_math(rendered: str, spans: list[tuple[str, bool]]) -> str:
    def put_back(match: re.Match[str]) -> str:
        index = int(match.group(1))
        if index >= len(spans):
            return match.group(0)
        latex, display = spans[index]
        mathml = _to_mathml(latex, display)
        if mathml is None:
            css = "math-raw math-block" if display else "math-raw"
            return f'<code class="{css}">{html.escape(latex)}</code>'
        return f'<span class="{"math-block" if display else "math"}">{mathml}</span>'

    return _PLACEHOLDER.sub(put_back, rendered)


def _wrap_tables(rendered: str) -> str:
    """Give every table its own scroll box; paper tables are far too wide."""
    return rendered.replace('<table>', '<div class="table-scroll"><table>').replace(
        "</table>", "</table></div>"
    )


def to_html(markdown: str, *, math: bool = True) -> str:
    source = markdown or ""
    if not math or "$" not in source:
        return _wrap_tables(_parser().render(source))
    parked, spans = _park_math(source)
    return _wrap_tables(_unpark_math(_parser().render(parked), spans))
