"""Turn a directory of LaTeX sources into one readable Markdown document.

The pipeline is: locate the root ``.tex`` file, splice in everything it
``\\input``s, expand the paper's own simple macros, then rewrite the LaTeX body
into Markdown. Brace matching is done by scanning rather than by regex, because
real papers nest commands several levels deep.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

log = logging.getLogger(__name__)

MAX_INPUT_DEPTH = 12

_ROOT_NAME_BONUS = ("main", "paper", "root", "ms", "arxiv", "manuscript", "article")
_ROOT_NAME_PENALTY = ("supp", "appendix", "sup", "response", "rebuttal", "cover")

# Environments whose content is noise once the prose is gone.
_DROP_ENVIRONMENTS = (
    "figure", "figure*", "wrapfigure", "subfigure", "SCfigure",
    "tikzpicture", "algorithm", "algorithmic", "algorithm2e",
    "thebibliography", "comment", "abstract*",
)
# Environments we unwrap, keeping the inner text.
_UNWRAP_ENVIRONMENTS = (
    "center", "small", "footnotesize", "scriptsize", "large", "Large",
    "flushleft", "flushright", "adjustbox", "spacing", "quote", "quotation",
    "minipage", "sloppypar", "leftbar",
)


# --------------------------------------------------------------------------
# Low-level scanning helpers
# --------------------------------------------------------------------------

def _read(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def strip_comments(text: str) -> str:
    """Remove ``%`` comments, honouring ``\\%`` and leaving verbatim blocks alone."""
    out: list[str] = []
    verbatim_depth = 0
    for line in text.splitlines():
        lowered = line.lstrip()
        if lowered.startswith((r"\begin{verbatim}", r"\begin{lstlisting}", r"\begin{minted}")):
            verbatim_depth += 1
        elif lowered.startswith((r"\end{verbatim}", r"\end{lstlisting}", r"\end{minted}")):
            verbatim_depth = max(0, verbatim_depth - 1)

        if verbatim_depth:
            out.append(line)
            continue

        cut = None
        escaped = False
        for index, char in enumerate(line):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == "%":
                cut = index
                break
        if cut is None:
            out.append(line)
        elif line[:cut].strip():
            out.append(line[:cut])
        # A comment-only line vanishes in TeX, newline included. Emitting a blank
        # line instead would fake a paragraph break, and a lone `%` between
        # sentences is a widespread idiom.
    return "\n".join(out)


def _match_brace(text: str, open_index: int) -> int:
    """Given the index of ``{``, return the index just past its matching ``}``."""
    depth = 0
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _read_group(text: str, index: int) -> tuple[str | None, int]:
    """Read a ``{...}`` group starting at or after ``index``, skipping whitespace."""
    cursor = index
    while cursor < len(text) and text[cursor] in " \t\n\r":
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None, index
    end = _match_brace(text, cursor)
    if end < 0:
        return None, index
    return text[cursor + 1 : end - 1], end


def _read_optional(text: str, index: int) -> tuple[str | None, int]:
    """Read a ``[...]`` optional argument starting at or after ``index``."""
    cursor = index
    while cursor < len(text) and text[cursor] in " \t":
        cursor += 1
    if cursor >= len(text) or text[cursor] != "[":
        return None, index
    depth = 0
    for position in range(cursor, len(text)):
        if text[position] == "[":
            depth += 1
        elif text[position] == "]":
            depth -= 1
            if depth == 0:
                return text[cursor + 1 : position], position + 1
    return None, index


def replace_command(
    text: str,
    name: str,
    nargs: int,
    render: Callable[[list[str], str | None], str],
    has_optional: bool = False,
) -> str:
    r"""Replace every ``\name{..}..`` occurrence using brace-aware scanning.

    ``render`` receives the required arguments and the optional one (or None).
    Occurrences with too few arguments are left untouched.
    """
    pattern = re.compile(r"\\" + re.escape(name) + r"(?![a-zA-Z@])")
    out: list[str] = []
    cursor = 0
    while True:
        match = pattern.search(text, cursor)
        if not match:
            out.append(text[cursor:])
            break
        out.append(text[cursor : match.start()])
        position = match.end()

        optional = None
        if has_optional:
            optional, position = _read_optional(text, position)

        args: list[str] = []
        ok = True
        for _ in range(nargs):
            arg, position = _read_group(text, position)
            if arg is None:
                ok = False
                break
            args.append(arg)

        if not ok:
            out.append(match.group(0))
            cursor = match.end()
            continue

        out.append(render(args, optional))
        cursor = position
    return "".join(out)


def _iter_environments(text: str, name: str) -> Iterable[tuple[int, int, int, int]]:
    r"""Yield ``(begin_start, body_start, body_end, end_stop)`` for one environment.

    Nested environments of the same name are handled, and the scan restarts
    after each match so overlapping is impossible.
    """
    begin_pattern = re.compile(r"\\begin\s*\{" + re.escape(name) + r"\}")
    end_pattern = re.compile(r"\\end\s*\{" + re.escape(name) + r"\}")
    cursor = 0
    while True:
        begin = begin_pattern.search(text, cursor)
        if not begin:
            return
        depth = 1
        position = begin.end()
        while depth:
            next_begin = begin_pattern.search(text, position)
            next_end = end_pattern.search(text, position)
            if not next_end:
                return
            if next_begin and next_begin.start() < next_end.start():
                depth += 1
                position = next_begin.end()
            else:
                depth -= 1
                position = next_end.end()
                if depth == 0:
                    yield begin.start(), begin.end(), next_end.start(), next_end.end()
        cursor = position


def _transform_environment(text: str, name: str, render: Callable[[str], str]) -> str:
    """Rewrite each ``\\begin{name}...\\end{name}`` block through ``render``."""
    while True:
        spans = list(_iter_environments(text, name))
        if not spans:
            return text
        begin_start, body_start, body_end, end_stop = spans[0]
        body = text[body_start:body_end]
        text = text[:begin_start] + render(body) + text[end_stop:]


# --------------------------------------------------------------------------
# Root discovery and \input splicing
# --------------------------------------------------------------------------

def _root_from_readme(source_dir: Path) -> Path | None:
    """arXiv ships a ``00README.json`` that names the top-level file. Trust it."""
    readme = source_dir / "00README.json"
    if not readme.is_file():
        return None
    try:
        manifest = json.loads(readme.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None
    for entry in manifest.get("sources") or []:
        if not isinstance(entry, dict) or entry.get("usage") != "toplevel":
            continue
        filename = entry.get("filename")
        if not filename:
            continue
        candidate = source_dir / filename
        if candidate.is_file():
            return candidate
    return None


def find_root_tex(source_dir: Path) -> Path | None:
    """Pick the ``.tex`` file that compiles the paper.

    Prefers arXiv's own declaration, otherwise scores candidates by whether they
    declare a document, how shallow they are, and whether the filename smells
    like a main file or a supplement.
    """
    declared = _root_from_readme(source_dir)
    if declared is not None:
        return declared

    candidates = [
        path for path in sorted(source_dir.rglob("*.tex"))
        if path.is_file() and not any(part.startswith(".") for part in path.parts)
    ]
    if not candidates:
        return None

    scored: list[tuple[float, Path]] = []
    for path in candidates:
        try:
            content = strip_comments(_read(path))
        except OSError:
            continue
        score = 0.0
        if r"\begin{document}" in content:
            score += 100
        if r"\documentclass" in content:
            score += 50
        if not score:
            continue

        stem = path.stem.lower()
        if any(token in stem for token in _ROOT_NAME_BONUS):
            score += 10
        if any(token in stem for token in _ROOT_NAME_PENALTY):
            score -= 40
        # Prefer files near the top of the tree.
        score -= 5 * len(path.relative_to(source_dir).parts)
        # Among equals, the one that pulls in more material is the real root.
        score += min(len(content) / 5000.0, 10)
        scored.append((score, path))

    if not scored:
        # No \begin{document} anywhere: fall back to the largest .tex file.
        return max(candidates, key=lambda p: p.stat().st_size)
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    return scored[0][1]


def _resolve_input_path(raw: str, base_dir: Path, source_dir: Path) -> Path | None:
    name = raw.strip().strip('"')
    if not name:
        return None
    name = name.replace("\\space", " ").strip()
    variants = [name] if name.lower().endswith((".tex", ".ltx")) else [f"{name}.tex", name]
    for directory in (base_dir, source_dir):
        for variant in variants:
            candidate = (directory / variant).resolve()
            try:
                candidate.relative_to(source_dir.resolve())
            except ValueError:
                continue  # Refuse to read outside the paper's own directory.
            if candidate.is_file():
                return candidate
    return None


def flatten_tex(root: Path, source_dir: Path) -> str:
    """Return the root file with every ``\\input``/``\\include`` spliced in."""
    visited: set[Path] = set()

    def expand(path: Path, depth: int) -> str:
        resolved = path.resolve()
        if depth > MAX_INPUT_DEPTH or resolved in visited:
            return ""
        visited.add(resolved)
        text = strip_comments(_read(path))

        for command in ("input", "include", "subfile", "InputIfFileExists"):
            def render(args: list[str], _optional: str | None, _cmd: str = command) -> str:
                target = _resolve_input_path(args[0], path.parent, source_dir)
                if target is None:
                    return ""
                return "\n" + expand(target, depth + 1) + "\n"

            text = replace_command(text, command, 1, render)

        # \import{dir}{file} and \subimport{dir}{file}
        for command in ("import", "subimport"):
            def render_import(args: list[str], _optional: str | None) -> str:
                target = _resolve_input_path(
                    str(Path(args[0]) / args[1]), path.parent, source_dir
                )
                if target is None:
                    return ""
                return "\n" + expand(target, depth + 1) + "\n"

            text = replace_command(text, command, 2, render_import)

        # Bare `\input filename` without braces.
        def bare_input(match: re.Match[str]) -> str:
            target = _resolve_input_path(match.group(1), path.parent, source_dir)
            return "\n" + expand(target, depth + 1) + "\n" if target else ""

        text = re.sub(r"\\input\s+([^\s{}\\]+)", bare_input, text)
        return text

    return expand(root, 0)


# --------------------------------------------------------------------------
# Macro expansion
# --------------------------------------------------------------------------

_MACRO_DEF = re.compile(
    r"\\(?:(?:new|renew|provide)command\*?\s*\{?|def\s*)\\([a-zA-Z@]+)\}?"
)


def collect_macros(text: str) -> dict[str, tuple[int, str]]:
    r"""Collect simple macro definitions as ``name -> (nargs, body)``.

    Handles ``\newcommand`` and the plain-TeX ``\def`` that many papers use for
    shorthands like ``\eg``. Only short, non-self-referential bodies are kept:
    the goal is to resolve those shorthands, not to reimplement TeX.
    """
    macros: dict[str, tuple[int, str]] = {}
    for match in _MACRO_DEF.finditer(text):
        name = match.group(1)
        position = match.end()
        nargs_raw, position = _read_optional(text, position)
        # A second optional argument means a default value, which we do not support.
        default, position = _read_optional(text, position)
        body, position = _read_group(text, position)
        if body is None or default is not None:
            continue
        try:
            nargs = int(nargs_raw) if nargs_raw else 0
        except ValueError:
            continue
        if nargs > 3 or len(body) > 300 or f"\\{name}" in body:
            continue
        macros[name] = (nargs, body)
    return macros


_DEFINITION_HEAD = re.compile(
    r"\\(newcommand|renewcommand|providecommand|def|edef|gdef|xdef|let|setlength"
    r"|DeclareMathOperator|newtheorem|newcolumntype|newenvironment|renewenvironment)"
    r"\*?(?![a-zA-Z@])"
)
# How many balanced groups follow the macro's name.
_DEFINITION_BODIES = {"newenvironment": 2, "renewenvironment": 2}


def _skip_definition(text: str, index: int, kind: str) -> int:
    """Return the offset just past a macro definition that starts at ``index``."""
    if kind == "let":
        match = re.compile(r"\s*\\[a-zA-Z@]+\s*=?\s*(?:\\[a-zA-Z@]+|.)").match(text, index)
        return match.end() if match else index

    if kind in ("def", "edef", "gdef", "xdef"):
        match = re.compile(r"\s*\\[a-zA-Z@]+").match(text, index)
        if not match:
            return index
        # Whatever sits between the name and the body is TeX's parameter text.
        cursor = text.find("{", match.end())
        if cursor < 0:
            return match.end()
        end = _match_brace(text, cursor)
        return end if end > 0 else match.end()

    cursor = index
    name, after = _read_group(text, cursor)
    if name is None:
        match = re.compile(r"\s*\\[a-zA-Z@]+").match(text, cursor)
        cursor = match.end() if match else cursor
    else:
        cursor = after

    while True:
        optional, after = _read_optional(text, cursor)
        if optional is None:
            break
        cursor = after

    for _ in range(_DEFINITION_BODIES.get(kind, 1)):
        body, after = _read_group(text, cursor)
        if body is None:
            break
        cursor = after
        while True:
            optional, after = _read_optional(text, cursor)
            if optional is None:
                break
            cursor = after
    return cursor


def drop_definitions(text: str) -> str:
    r"""Delete macro definitions themselves.

    Must run after :func:`collect_macros` and before :func:`expand_macros`, or a
    definition like ``\def\cA{\mathcal{A}}`` gets rewritten into the nonsense
    ``\def\mathcal{A}{\mathcal{A}}`` when its own shorthand is substituted.
    """
    out: list[str] = []
    cursor = 0
    while True:
        match = _DEFINITION_HEAD.search(text, cursor)
        if not match:
            out.append(text[cursor:])
            return "".join(out)
        out.append(text[cursor : match.start()])
        cursor = max(_skip_definition(text, match.end(), match.group(1)), match.end())


def expand_macros(text: str, macros: dict[str, tuple[int, str]], passes: int = 3) -> str:
    """Substitute collected macros, a few passes deep for nested shorthands."""
    if not macros:
        return text
    # Longest names first so \methodname is not eaten by \method.
    names = sorted(macros, key=len, reverse=True)
    for _ in range(passes):
        before = text
        for name in names:
            nargs, body = macros[name]

            def render(args: list[str], _optional: str | None, _body: str = body) -> str:
                result = _body
                for position, value in enumerate(args, start=1):
                    result = result.replace(f"#{position}", value)
                return result

            text = replace_command(text, name, nargs, render)
        if text == before:
            break
    return text


# --------------------------------------------------------------------------
# Bibliography
# --------------------------------------------------------------------------

_BIBITEM = re.compile(r"\\bibitem(?:\[[^\]]*\])?\s*\{([^}]+)\}")
_BIB_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.IGNORECASE)
_BIB_FIELD = re.compile(r"(\w+)\s*=\s*", re.IGNORECASE)
# DOIs contain digit groups that look exactly like modern arXiv ids
# (10.1109/CVPR.2023.12345), so they are removed before any id hunting.
_DOI_LIKE = re.compile(r"\b10\.\d{4,}/\S+")
_NEAR_ARXIV = re.compile(r"arxiv[^\n]{0,40}", re.IGNORECASE)


def _clean_bib_text(text: str) -> str:
    text = re.sub(r"\\[a-zA-Z@]+\s*", " ", text)
    text = re.sub(r"[{}~\\]", " ", text)
    return " ".join(text.split())


@dataclass
class BibEntry:
    """One bibliography entry, however the paper happened to store it."""

    key: str
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    arxiv_id: str | None = None
    doi: str | None = None
    raw: str = ""

    @property
    def label(self) -> str:
        if not self.authors:
            # Without a parsed author list the rendered text is the best we have,
            # and it at least starts with the authors.
            return self.raw[:180]
        parts = [part for part in (self.authors, self.year, self.title) if part]
        return ", ".join(parts)[:180]

    @property
    def display(self) -> str:
        title = self.title or self.raw[:100]
        first_author = self.authors.split(" and ")[0].strip()
        bits = [bit for bit in (first_author, self.year) if bit]
        return f"{title} ({', '.join(bits)})" if bits else title


def _arxiv_id_from_text(text: str) -> str | None:
    """Find an arXiv id, but only where the text says it is one."""
    from .resolve import extract_refs

    cleaned = _DOI_LIKE.sub(" ", text)
    # Requiring the id to sit right after the word "arXiv" avoids matching
    # conference numbering, page ranges and DOI fragments.
    for window in _NEAR_ARXIV.findall(cleaned):
        refs = extract_refs(window)
        if refs:
            return refs[0].canonical
    return None


def _entry_from_fields(key: str, fields: dict[str, str], raw: str) -> BibEntry:
    from .resolve import extract_refs

    arxiv_id = None
    eprint = fields.get("eprint", "") or fields.get("arxivid", "")
    archive = (fields.get("archiveprefix", "") or "").lower()
    if eprint and ("arxiv" in archive or not archive or "arxiv" in eprint.lower()):
        refs = extract_refs(eprint)
        arxiv_id = refs[0].canonical if refs else None
    if arxiv_id is None:
        searchable = " ".join(
            fields.get(name, "")
            for name in ("journal", "booktitle", "url", "note", "howpublished", "series")
        )
        arxiv_id = _arxiv_id_from_text(searchable) or _arxiv_id_from_text(raw)

    return BibEntry(
        key=key,
        title=fields.get("title", ""),
        authors=fields.get("author", ""),
        year=fields.get("year", "") or _year_in(raw),
        venue=fields.get("journal", "") or fields.get("booktitle", ""),
        arxiv_id=arxiv_id,
        doi=fields.get("doi") or None,
        raw=raw[:400],
    )


def _year_in(text: str) -> str:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else ""


# A .bbl entry has no fields, but both common styles mark the title
# structurally: natbib puts it after the first \newblock, IEEE quotes it.
_NEWBLOCK = re.compile(r"\\newblock\s*")
_TEX_QUOTED = re.compile(r"``(.+?)''", re.DOTALL)
_TRAILING_YEAR = re.compile(r",?\s*(19|20)\d{2}\s*\.?$")


def _title_from_bbl(chunk: str) -> str:
    quoted = _TEX_QUOTED.search(chunk)
    if quoted:
        return _clean_bib_text(quoted.group(1)).strip(" ,.")
    blocks = _NEWBLOCK.split(chunk)
    if len(blocks) > 1:
        return _TRAILING_YEAR.sub("", _clean_bib_text(blocks[1])).strip(" ,.")
    return ""


def _authors_from_bbl(chunk: str) -> str:
    """Whatever precedes the title, which in both styles is the author list."""
    quoted = _TEX_QUOTED.search(chunk)
    if quoted:
        return _clean_bib_text(chunk[: quoted.start()]).strip(" ,.")
    blocks = _NEWBLOCK.split(chunk)
    return _clean_bib_text(blocks[0]).strip(" ,.") if len(blocks) > 1 else ""


def _bib_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for match in _BIB_FIELD.finditer(body):
        value, _ = _read_group(body, match.end())
        if value is None:
            quoted = re.match(r'\s*"([^"]*)"', body[match.end() :])
            value = quoted.group(1) if quoted else ""
        fields[match.group(1).lower()] = _clean_bib_text(value)
    return fields


def parse_bib_entries(source_dir: Path) -> dict[str, BibEntry]:
    """Parse every bibliography in a source tree, keyed by citation key.

    ``.bbl`` files are read first because they are what actually got compiled,
    then ``.bib`` sources fill in the structured fields that a ``.bbl`` has
    already flattened away.
    """
    entries: dict[str, BibEntry] = {}

    for path in sorted(source_dir.rglob("*.bbl")):
        content = strip_comments(_read(path))
        matches = list(_BIBITEM.finditer(content))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
            chunk = content[match.end() : end]
            raw = _clean_bib_text(chunk)
            if not raw:
                continue
            key = match.group(1).strip()
            entries.setdefault(
                key,
                BibEntry(
                    key=key,
                    title=_title_from_bbl(chunk),
                    authors=_authors_from_bbl(chunk),
                    year=_year_in(raw),
                    arxiv_id=_arxiv_id_from_text(chunk),
                    raw=raw[:400],
                ),
            )

    for path in sorted(source_dir.rglob("*.bib")):
        content = strip_comments(_read(path))
        found = list(_BIB_ENTRY.finditer(content))
        for index, match in enumerate(found):
            key = match.group(2).strip()
            end = found[index + 1].start() if index + 1 < len(found) else len(content)
            body = content[match.end() : end]
            parsed = _entry_from_fields(key, _bib_fields(body), _clean_bib_text(body))
            existing = entries.get(key)
            if existing is None:
                entries[key] = parsed
                continue
            # The .bbl entry wins on text, the .bib entry supplies the metadata.
            existing.title = existing.title or parsed.title
            existing.authors = existing.authors or parsed.authors
            existing.year = existing.year or parsed.year
            existing.venue = existing.venue or parsed.venue
            existing.arxiv_id = existing.arxiv_id or parsed.arxiv_id
            existing.doi = existing.doi or parsed.doi

    return entries


def parse_bibliography(source_dir: Path) -> dict[str, str]:
    """Map citation keys to a short ``Author, Year, Title`` label."""
    return {key: entry.label for key, entry in parse_bib_entries(source_dir).items()}


def _short_citation(key: str, labels: dict[str, str]) -> str:
    key = key.strip()
    label = labels.get(key)
    if not label:
        return key
    # "Vaswani, Ashish and ..., 2017, Attention is all you need" -> "Vaswani 2017"
    # "A. Ramesh, M. Pavlov, ..., 2021, Zero-shot ..."           -> "Ramesh 2021"
    # "Kaiming He and others. Deep residual learning, 2016."     -> "He 2016"
    head = label.split(",")[0]
    # Only break at a real sentence end: the lookbehind spares initials like "A.".
    head = re.split(r"(?<=[a-z])\.\s", head)[0]
    head = re.split(r"\band\b", head)[0]
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", head)
    surname = tokens[-1] if tokens else ""
    year = re.search(r"\b(19|20)\d{2}\b", label)
    if not surname:
        return f"{key} {year.group(0)}" if year else key
    return f"{surname} {year.group(0)}" if year else surname


# --------------------------------------------------------------------------
# LaTeX -> Markdown
# --------------------------------------------------------------------------

_SECTION_LEVELS = {
    "part": "#",
    "chapter": "#",
    "section": "##",
    "subsection": "###",
    "subsubsection": "####",
    "paragraph": "#####",
    "subparagraph": "#####",
}

_DROP_COMMANDS_NOARG = (
    "noindent", "centering", "clearpage", "newpage", "bigskip", "medskip",
    "smallskip", "hline", "toprule", "midrule", "bottomrule", "hfill", "vfill",
    "linebreak", "par", "raggedright", "raggedleft", "footnotesize", "small",
    "scriptsize", "tiny", "normalsize", "large", "Large", "LARGE", "huge",
    "Huge", "bfseries", "itshape", "ttfamily", "maketitle", "tableofcontents",
    "listoffigures", "listoftables", "hfil", "quad", "qquad", "columnbreak",
    "xspace", "protect", "boldmath", "unboldmath", "em", "rm", "sf", "tt",
    "sl", "sc", "it", "normalfont", "leavevmode", "relax", "topsep",
    "bf", "pagebreak", "nobreak", "allowbreak", "unskip", "strut",
    "FloatBarrier", "endinput", "appendix", "fill", "onedot", "makeatletter",
    "makeatother", "sloppy", "flushbottom", "twocolumn", "onecolumn",
)

_DROP_COMMANDS_ONEARG = (
    "label", "vspace", "hspace", "includegraphics", "setlength", "bibliographystyle",
    "bibliography", "usepackage", "documentclass", "pagestyle", "thispagestyle",
    "addcontentsline", "captionsetup", "definecolor", "graphicspath", "input@path",
    "renewcommand", "newcommand", "providecommand", "DeclareMathOperator",
    "setcounter", "addtocounter", "arraystretch", "resizebox", "phantom",
    "color", "rowcolor", "cellcolor", "columncolor", "arrayrulecolor",
    "pagecolor", "colorbox", "keywords", "authorrunning", "titlerunning",
    "ding", "arabic", "roman", "alph", "vcenter", "hbox",
)

_TEXT_STYLE = {
    "textbf": "**", "bf": "**", "textit": "*", "emph": "*", "textsc": "",
    "textrm": "", "textsf": "", "textnormal": "", "text": "", "mbox": "",
    "underline": "", "uline": "", "textsuperscript": "", "textsubscript": "",
}

_SYMBOLS = {
    r"\ ": " ", r"\,": " ", r"\;": " ", r"\:": " ",
    r"\!": "", r"~": " ", r"\@": "", r"\/": "",
    r"\ldots": "...", r"\dots": "...", r"\cdots": "...",
    r"\textquotedblleft": '"', r"\textquotedblright": '"',
    r"\textasciitilde": "~", r"\textbackslash": "\\",
    r"``": '"', r"''": '"',
}

# Escaped literals are parked on private-use code points so that later brace
# stripping and math detection cannot mistake them for markup.
_ESCAPES = {
    r"\%": "\ue000", r"\&": "\ue001", r"\_": "\ue002", r"\#": "\ue003",
    r"\$": "\ue004", r"\{": "\ue005", r"\}": "\ue006",
}
_UNESCAPES = {
    "\ue000": "%", "\ue001": "&", "\ue002": "_", "\ue003": "#",
    "\ue004": "$", "\ue005": "{", "\ue006": "}",
}

def _strip_group_braces(text: str) -> str:
    r"""Drop grouping braces while keeping the ones that are command arguments.

    ``{\bfseries important}`` is a group and loses its braces, but ``\mathcal{L}``
    and ``x^{2n}`` must keep theirs or the math becomes wrong. Deciding by what
    precedes the brace is more reliable than trying to find the math spans, which
    can be delimited four different ways and can straddle lines.
    """
    out: list[str] = []
    after_command = False
    last_char = ""
    index = 0
    length = len(text)

    while index < length:
        char = text[index]

        if char == "\\" and index + 1 < length:
            end = index + 1
            if text[end].isalpha():
                while end < length and text[end].isalpha():
                    end += 1
            else:
                end = index + 2  # an escaped symbol such as \, or \|
            out.append(text[index:end])
            after_command = text[index + 1].isalpha()
            last_char = text[end - 1]
            index = end
            continue

        if char == "{":
            close = _match_brace(text, index)
            if close < 0:
                index += 1
                continue
            argument_of_command = after_command
            if argument_of_command or last_char in ("_", "^"):
                out.append(text[index:close])
                last_char = "}"
            else:
                # Recurse so nested groups inside a discarded group also go.
                out.append(_strip_group_braces(text[index + 1 : close - 1]))
                last_char = out[-1][-1] if out[-1] else last_char
            # Stay armed so the second group of \frac{a}{b} is kept as well.
            after_command = argument_of_command
            index = close
            continue

        if char == "}":
            index += 1
            continue

        if char not in " \t":
            after_command = False
            last_char = char
        out.append(char)
        index += 1

    return "".join(out)


def _reflow(text: str) -> str:
    """Rejoin hard-wrapped prose into paragraphs; LaTeX treats a newline as a space.

    Structured blocks (headings, lists, tables, display math) keep their lines.
    """
    blocks: list[str] = []
    for block in re.split(r"\n{2,}", text):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if any(line.startswith("#") or " | " in line for line in lines):
            blocks.append("\n".join(lines))
        elif any(line.startswith("- ") for line in lines):
            items: list[str] = []
            for line in lines:
                if line.startswith("- ") or not items:
                    items.append(line)
                else:
                    items[-1] = f"{items[-1]} {line}"
            blocks.append("\n".join(items))
        else:
            blocks.append(" ".join(lines))
    return "\n\n".join(blocks)


def _render_caption(body: str) -> str:
    caption = ""

    def grab(args: list[str], optional: str | None) -> str:
        nonlocal caption
        if not caption:
            caption = args[0]
        return ""

    replace_command(body, "caption", 1, grab, has_optional=True)
    return caption


_TABULAR_ENVIRONMENTS = ("tabular", "tabularx", "tabu", "longtable", "array")
# Very large tables cost a lot of context for little added meaning.
_MAX_TABLE_ROWS = 80


def _tabular_rows(inner: str) -> list[str]:
    """Flatten the body of a ``tabular`` into pipe-separated rows."""
    # Drop the column specification that follows \begin{tabular}.
    spec, offset = _read_group(inner, 0)
    if spec is not None and re.fullmatch(r"[lcrpXm@{}|\d.\s>%\\]*", spec or ""):
        inner = inner[offset:]

    rows: list[str] = []
    for line in re.split(r"\\\\", inner):
        cells = [" ".join(cell.split()) for cell in re.split(r"(?<!\\)&", line)]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _render_table(body: str) -> str:
    """Render a table float: keep the caption and the numbers, drop the layout."""
    parts: list[str] = []
    caption = _render_caption(body)
    if caption:
        parts.append(f"**Table.** {caption}")
    for env in _TABULAR_ENVIRONMENTS:
        for _begin, body_start, body_end, _end in _iter_environments(body, env):
            parts.extend(_tabular_rows(body[body_start:body_end]))
    parts = parts[: _MAX_TABLE_ROWS + 1]
    return "\n\n" + "\n".join(parts) + "\n\n" if parts else "\n"


def _render_bare_tabular(body: str) -> str:
    """Render a ``tabular`` that was not wrapped in a table float."""
    rows = _tabular_rows(body)[:_MAX_TABLE_ROWS]
    return "\n\n" + "\n".join(rows) + "\n\n" if rows else "\n"


def _render_figure(body: str) -> str:
    caption = _render_caption(body)
    return f"\n\n**Figure.** {caption}\n\n" if caption else "\n"


def tex_to_markdown(tex: str, bib_labels: dict[str, str] | None = None) -> str:
    """Rewrite a flattened LaTeX body into Markdown-ish plain text."""
    labels = bib_labels or {}

    # Figures and floats first: their captions are worth keeping, the rest is not.
    for env in ("figure*", "figure", "wrapfigure", "SCfigure"):
        tex = _transform_environment(tex, env, _render_figure)
    for env in ("table*", "table", "sidewaystable"):
        tex = _transform_environment(tex, env, _render_table)
    # Any tabular left over was not inside a float.
    for env in _TABULAR_ENVIRONMENTS:
        tex = _transform_environment(tex, env, _render_bare_tabular)

    for env in _DROP_ENVIRONMENTS:
        tex = _transform_environment(tex, env, lambda _body: "\n")
    for env in _UNWRAP_ENVIRONMENTS:
        tex = _transform_environment(tex, env, lambda body: body)

    tex = _transform_environment(tex, "abstract", lambda body: f"\n\n## Abstract\n\n{body}\n\n")
    for env in ("equation", "equation*", "align", "align*", "gather", "gather*", "eqnarray", "multline"):
        tex = _transform_environment(tex, env, lambda body: f"\n\n$$ {' '.join(body.split())} $$\n\n")

    # Lists.
    for env in ("itemize", "enumerate", "description", "compactitem"):
        tex = _transform_environment(tex, env, lambda body: f"\n{body}\n")
    tex = re.sub(r"\\item\s*\[([^\]]*)\]", r"\n- **\1** ", tex)
    tex = re.sub(r"\\item\b", "\n- ", tex)

    # Sections.
    for name, hashes in _SECTION_LEVELS.items():
        def render_section(args: list[str], _optional: str | None, _h: str = hashes) -> str:
            return f"\n\n{_h} {' '.join(args[0].split())}\n\n"

        tex = replace_command(tex, name, 1, render_section, has_optional=True)
        tex = replace_command(tex, f"{name}*", 1, render_section, has_optional=True)

    # Citations and cross-references.
    for name in ("cite", "citep", "citet", "citealp", "citealt", "citeauthor",
                 "citeyear", "Citep", "Citet", "parencite", "textcite", "autocite"):
        def render_cite(args: list[str], _optional: str | None) -> str:
            keys = [key.strip() for key in args[0].split(",") if key.strip()]
            return "[" + "; ".join(_short_citation(key, labels) for key in keys) + "]"

        tex = replace_command(tex, name, 1, render_cite, has_optional=True)

    for name in ("ref", "eqref", "autoref", "Cref", "cref", "pageref", "nameref"):
        tex = replace_command(tex, name, 1, lambda args, _o: f"({args[0]})")

    tex = replace_command(tex, "footnote", 1, lambda args, _o: f" ({args[0]})", has_optional=True)
    tex = replace_command(tex, "href", 2, lambda args, _o: f"{args[1]} ({args[0]})")
    tex = replace_command(tex, "url", 1, lambda args, _o: args[0])
    tex = replace_command(tex, "texttt", 1, lambda args, _o: f"`{args[0]}`")
    tex = replace_command(tex, "title", 1, lambda args, _o: f"\n\n# {args[0]}\n\n", has_optional=True)

    for name, marker in _TEXT_STYLE.items():
        tex = replace_command(tex, name, 1, lambda args, _o, m=marker: f"{m}{args[0]}{m}")

    tex = replace_command(tex, "textcolor", 2, lambda args, _o: args[1])

    # Table and box scaffolding whose last argument is the content we want.
    tex = replace_command(tex, "multicolumn", 3, lambda args, _o: args[2])
    tex = replace_command(tex, "rule", 2, lambda _args, _o: " ", has_optional=True)
    tex = replace_command(tex, "specialrule", 3, lambda _args, _o: " ")
    tex = re.sub(r"\\c(?:line|midrule)\s*(?:\([^)]*\))?\s*(?:\{[^}]*\})?", " ", tex)
    # \multirow{2}{*}{text} and the \multirow{2}*{text} spelling both occur.
    tex = re.sub(r"\\multirow\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}\s*(?:\*|\{[^{}]*\})\s*", "", tex)

    for name in ("shortstack", "makecell", "thead", "smash", "fbox", "framebox", "text"):
        tex = replace_command(tex, name, 1, lambda args, _o: args[0], has_optional=True)
    for name in ("raisebox", "scalebox", "rotatebox", "parbox", "tabincell", "adjustbox"):
        tex = replace_command(tex, name, 2, lambda args, _o: args[1], has_optional=True)

    for name in _DROP_COMMANDS_ONEARG:
        tex = replace_command(tex, name, 1, lambda _args, _o: "", has_optional=True)
    for name in _DROP_COMMANDS_NOARG:
        tex = re.sub(r"\\" + name + r"(?![a-zA-Z@])\s*", " ", tex)

    tex = re.sub(r"\\(begin|end)\s*\{[^}]*\}", "", tex)

    for source, sentinel in _ESCAPES.items():
        tex = tex.replace(source, sentinel)
    for source, target in _SYMBOLS.items():
        tex = tex.replace(source, target)
    tex = re.sub(r"\\\\\s*", "\n", tex)
    tex = _strip_group_braces(tex)
    for sentinel, target in _UNESCAPES.items():
        tex = tex.replace(sentinel, target)

    # Collapse whitespace without destroying paragraph structure.
    tex = re.sub(r"[ \t]+", " ", tex)
    tex = re.sub(r" *\n *", "\n", tex)
    tex = re.sub(r"\n{3,}", "\n\n", tex)
    return _reflow(tex.strip())


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split rendered Markdown into ``(heading, body)`` pairs at ``##`` level."""
    sections: list[tuple[str, str]] = []
    heading = "Preamble"
    buffer: list[str] = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if match and len(match.group(1)) <= 2:
            if any(item.strip() for item in buffer):
                sections.append((heading, "\n".join(buffer).strip()))
            heading = match.group(2).strip() or "Untitled"
            buffer = []
        else:
            buffer.append(line)
    if any(item.strip() for item in buffer):
        sections.append((heading, "\n".join(buffer).strip()))
    return sections


# --------------------------------------------------------------------------
# Structure and citation placement
# --------------------------------------------------------------------------

_HEADING_LEVELS = {
    "part": 1, "chapter": 1, "section": 1,
    "subsection": 2, "subsubsection": 3, "paragraph": 4,
}
_CITE_COMMANDS = re.compile(
    r"cite[a-zA-Z]*|parencite|textcite|autocite|nocite|footcite", re.IGNORECASE
)
_STRUCTURE = re.compile(
    r"\\(part|chapter|section|subsection|subsubsection|paragraph)\*?(?![a-zA-Z@])"
    r"|\\(cite[a-zA-Z]*|parencite|textcite|autocite|nocite|footcite)(?![a-zA-Z@])"
    r"|\\(begin|end)(?![a-zA-Z@])"
)
# Surveys put sprawling comparison tables inside whichever subsection they happen
# to fit near, so citations from a float say little about the taxonomy.
_FLOAT_ENVS = {
    "table", "table*", "figure", "figure*", "tabular", "tabular*", "tabularx",
    "longtable", "wraptable", "wrapfigure", "subtable", "subfigure", "tabu",
    "threeparttable", "sidewaystable",
}


@dataclass
class Heading:
    level: int
    title: str

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{'  ' * (self.level - 1)}{self.title}"


@dataclass
class CitationUse:
    """Where and how often one bibliography key is cited."""

    key: str
    # Citations in running prose, which is the signal worth trusting.
    count: int = 0
    # Citations from inside tables and figures, counted but kept apart.
    table_count: int = 0
    # Prose citations per section path, e.g. {"Methods > Diffusion-based": 3}.
    sections: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.count + self.table_count

    def add(self, section_path: str, in_float: bool = False) -> None:
        if in_float:
            self.table_count += 1
            return
        self.count += 1
        if section_path:
            self.sections[section_path] = self.sections.get(section_path, 0) + 1


def _clean_inline(text: str) -> str:
    """Reduce a heading or short fragment to plain readable text."""
    text = replace_command(text, "texorpdfstring", 2, lambda args, _o: args[1])
    text = re.sub(r"\\(?:label|thanks|footnote)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z@]+\s*", " ", text)
    text = re.sub(r"[{}$\\]", " ", text)
    return " ".join(text.split())


def _walk_structure(tex: str) -> Iterable[tuple[str, str, int]]:
    """Yield ``(kind, content, level)`` for headings and citations, in order."""
    cursor = 0
    while True:
        match = _STRUCTURE.search(tex, cursor)
        if not match:
            return
        position = match.end()
        # Both heading and citation commands can carry optional arguments.
        while True:
            optional, after = _read_optional(tex, position)
            if optional is None:
                break
            position = after
        content, after = _read_group(tex, position)
        if content is None:
            cursor = match.end()
            continue
        if match.group(1):
            yield "heading", content, _HEADING_LEVELS.get(match.group(1), 3)
        elif match.group(2):
            yield "cite", content, 0
        else:
            yield match.group(3), content.strip(), 0
        cursor = after


def _flatten_for_scan(source_dir: Path) -> str:
    root = find_root_tex(source_dir)
    if root is None:
        return ""
    flattened = flatten_tex(root, source_dir)
    body_start = flattened.find(r"\begin{document}")
    body = flattened[body_start:] if body_start >= 0 else flattened
    return drop_definitions(body)


def section_tree(source_dir: Path) -> list[Heading]:
    """The paper's heading hierarchy.

    For a survey this is the most valuable structure in the document: the way the
    authors carved up the field, stated by them rather than inferred by a model.
    """
    headings: list[Heading] = []
    for kind, content, level in _walk_structure(_flatten_for_scan(source_dir)):
        if kind != "heading":
            continue
        title = _clean_inline(content)
        if title:
            headings.append(Heading(level=level, title=title))
    return headings


def scan_citations(source_dir: Path) -> dict[str, CitationUse]:
    """Count every citation and record which sections it appears in.

    Knowing that a reference is cited under "3.2 Diffusion-based methods" places
    it in the survey's own taxonomy without asking a model to guess.
    """
    uses: dict[str, CitationUse] = {}
    path: list[str] = []
    float_depth = 0

    for kind, content, level in _walk_structure(_flatten_for_scan(source_dir)):
        if kind == "heading":
            title = _clean_inline(content)
            if not title or level > 3:
                continue
            del path[level - 1 :]
            path.append(title)
            continue

        if kind == "begin":
            float_depth += content in _FLOAT_ENVS
            continue
        if kind == "end":
            float_depth = max(float_depth - (content in _FLOAT_ENVS), 0)
            continue

        section_path = " > ".join(path)
        for raw_key in content.split(","):
            key = raw_key.strip()
            if not key or key == "*":
                continue
            uses.setdefault(key, CitationUse(key=key)).add(
                section_path, in_float=float_depth > 0
            )
    return uses


def build_fulltext(source_dir: Path) -> tuple[str, Path | None]:
    """Convert a paper's source tree to Markdown. Returns ``(text, root_file)``."""
    root = find_root_tex(source_dir)
    if root is None:
        return "", None

    flattened = flatten_tex(root, source_dir)
    macros = collect_macros(flattened)

    body_start = flattened.find(r"\begin{document}")
    preamble = flattened[:body_start] if body_start >= 0 else ""
    body = flattened[body_start:] if body_start >= 0 else flattened
    # The title usually lives in the preamble; keep it and discard the rest.
    title_match = re.search(r"\\title\s*(?:\[[^\]]*\])?\s*\{", preamble)
    if title_match:
        end = _match_brace(preamble, title_match.end() - 1)
        if end > 0:
            body = preamble[title_match.start() : end] + "\n" + body

    body = expand_macros(drop_definitions(body), macros)
    return tex_to_markdown(body, parse_bibliography(source_dir)), root
