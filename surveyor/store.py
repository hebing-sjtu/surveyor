"""On-disk paper library.

Everything is plain files so the library stays greppable, diffable and
git-friendly::

    papers/<paper_id>/
        source/        raw LaTeX tree as downloaded from arXiv
        meta.json      bibliographic record
        fulltext.md    flattened, de-TeX-ed body
        summary.json   structured note
        summary.md     the same note, rendered for humans
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Iterator

from .config import Settings, get_settings
from .ingest.tex import build_fulltext, split_sections
from .models import (
    Chunk,
    PaperMeta,
    PaperRecord,
    PaperSummary,
    SurveyNote,
    SurveyReference,
)

log = logging.getLogger(__name__)

META_FILE = "meta.json"
FULLTEXT_FILE = "fulltext.md"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"
SURVEY_JSON = "survey.json"
SURVEY_MD = "survey.md"
REFERENCES_JSON = "references.json"
SOURCE_DIR = "source"

# Sections that add length but little meaning for summarization and retrieval.
_SKIP_SECTIONS = re.compile(
    r"^(acknowledg|references|bibliograph|appendix\s*$|author contribution|"
    r"funding|conflict of interest|checklist|neurips paper checklist)",
    re.IGNORECASE,
)


class PaperStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()

    # ---------------------------------------------------------------- paths
    @property
    def root(self) -> Path:
        return self.settings.papers_dir

    def paper_dir(self, paper_id: str) -> Path:
        return self.root / paper_id

    def source_dir(self, paper_id: str) -> Path:
        return self.paper_dir(paper_id) / SOURCE_DIR

    def exists(self, paper_id: str) -> bool:
        return (self.paper_dir(paper_id) / META_FILE).is_file()

    def list_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / META_FILE).is_file()
        )

    # ----------------------------------------------------------------- meta
    def load_meta(self, paper_id: str) -> PaperMeta | None:
        path = self.paper_dir(paper_id) / META_FILE
        if not path.is_file():
            return None
        try:
            return PaperMeta.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("could not read %s: %s", path, exc)
            return None

    def save_meta(self, meta: PaperMeta) -> None:
        directory = self.paper_dir(meta.paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / META_FILE).write_text(
            meta.model_dump_json(indent=2), encoding="utf-8"
        )

    # -------------------------------------------------------------- content
    def load_fulltext(self, paper_id: str) -> str:
        path = self.paper_dir(paper_id) / FULLTEXT_FILE
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def save_fulltext(self, paper_id: str, text: str) -> None:
        directory = self.paper_dir(paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / FULLTEXT_FILE).write_text(text, encoding="utf-8")

    def rebuild_fulltext(self, paper_id: str) -> str:
        """Re-run the TeX conversion from the stored source tree."""
        source = self.source_dir(paper_id)
        if not source.is_dir():
            return ""
        text, root = build_fulltext(source)
        if not text:
            log.warning("%s: no usable LaTeX source found", paper_id)
            return ""
        log.debug("%s: converted %s (%d chars)", paper_id, root, len(text))
        self.save_fulltext(paper_id, text)
        return text

    # -------------------------------------------------------------- summary
    def load_summary(self, paper_id: str) -> PaperSummary | None:
        path = self.paper_dir(paper_id) / SUMMARY_JSON
        if not path.is_file():
            return None
        try:
            return PaperSummary.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("could not read %s: %s", path, exc)
            return None

    def save_summary(self, summary: PaperSummary, rendered: str) -> None:
        directory = self.paper_dir(summary.paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SUMMARY_JSON).write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / SUMMARY_MD).write_text(rendered, encoding="utf-8")

    # ---------------------------------------------------------------- survey
    def list_surveys(self) -> list[str]:
        return [
            paper_id
            for paper_id in self.list_ids()
            if (meta := self.load_meta(paper_id)) and meta.kind == "survey"
        ]

    def load_survey(self, paper_id: str) -> SurveyNote | None:
        path = self.paper_dir(paper_id) / SURVEY_JSON
        if not path.is_file():
            return None
        try:
            return SurveyNote.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("could not read %s: %s", path, exc)
            return None

    def save_survey(self, note: SurveyNote, rendered: str) -> None:
        directory = self.paper_dir(note.paper_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SURVEY_JSON).write_text(
            note.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / SURVEY_MD).write_text(rendered, encoding="utf-8")

    def load_references(self, paper_id: str) -> list[SurveyReference]:
        payload = load_json(self.paper_dir(paper_id) / REFERENCES_JSON, [])
        if not isinstance(payload, list):
            return []
        references: list[SurveyReference] = []
        for item in payload:
            try:
                references.append(SurveyReference.model_validate(item))
            except Exception:
                continue
        return references

    def save_references(self, paper_id: str, references: list[SurveyReference]) -> None:
        save_json(
            self.paper_dir(paper_id) / REFERENCES_JSON,
            [reference.model_dump() for reference in references],
        )

    # --------------------------------------------------------------- records
    def load_record(self, paper_id: str) -> PaperRecord | None:
        meta = self.load_meta(paper_id)
        if meta is None:
            return None
        return PaperRecord(
            meta=meta,
            summary=self.load_summary(paper_id),
            fulltext_chars=len(self.load_fulltext(paper_id)),
        )

    def iter_records(self) -> Iterator[PaperRecord]:
        for paper_id in self.list_ids():
            record = self.load_record(paper_id)
            if record is not None:
                yield record

    def delete(self, paper_id: str) -> bool:
        directory = self.paper_dir(paper_id)
        if not directory.is_dir():
            return False
        shutil.rmtree(directory)
        return True

    # -------------------------------------------------------------- lookup
    def resolve(self, query: str) -> list[str]:
        """Find papers by id fragment, title words, or tag.

        Returns every match so callers can disambiguate; an exact id match short
        circuits to a single result.
        """
        query = query.strip()
        if not query:
            return []
        if self.exists(query):
            return [query]

        needle = query.lower()
        exact_arxiv: list[str] = []
        partial: list[str] = []
        for record in self.iter_records():
            meta = record.meta
            if meta.arxiv_id and needle == meta.arxiv_id.lower():
                exact_arxiv.append(meta.paper_id)
            elif needle in meta.paper_id.lower() or needle in meta.title.lower() or any(needle == tag.lower() for tag in meta.tags):
                partial.append(meta.paper_id)
        return exact_arxiv or partial

    def resolve_one(self, query: str) -> str | None:
        matches = self.resolve(query)
        return matches[0] if len(matches) == 1 else None

    # -------------------------------------------------------------- chunking
    def chunks(self, paper_id: str, target_chars: int = 4000) -> list[Chunk]:
        """Split a paper into retrieval-sized pieces, one or more per section."""
        text = self.load_fulltext(paper_id)
        if not text:
            return []

        chunks: list[Chunk] = []
        for heading, body in split_sections(text):
            if _SKIP_SECTIONS.match(heading.strip()):
                continue
            for index, piece in enumerate(_split_paragraphs(body, target_chars)):
                chunks.append(
                    Chunk(
                        paper_id=paper_id,
                        chunk_id=f"{paper_id}::{len(chunks):03d}",
                        section=heading if index == 0 else f"{heading} ({index + 1})",
                        text=piece,
                    )
                )
        return chunks


def _split_paragraphs(body: str, target_chars: int) -> list[str]:
    """Group paragraphs into pieces of roughly ``target_chars``, never mid-paragraph."""
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for paragraph in body.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if size + len(paragraph) > target_chars and current:
            pieces.append("\n\n".join(current))
            current, size = [], 0
        current.append(paragraph)
        size += len(paragraph) + 2
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def write_index(store: PaperStore) -> Path:
    """Refresh ``knowledge/index.md``, the human-facing table of contents."""
    records = sorted(
        store.iter_records(),
        key=lambda record: (record.meta.published or "", record.meta.paper_id),
        reverse=True,
    )
    surveys = sum(1 for record in records if record.meta.kind == "survey")
    lines = [
        "# Reading list",
        "",
        f"{len(records)} entries, {surveys} of them surveys.",
        "",
        "| Paper | Kind | Title | Note | Topics |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in records:
        meta, summary = record.meta, record.summary
        note_file = "survey.md" if meta.kind == "survey" else "summary.md"
        link = f"[{meta.paper_id}](../papers/{meta.paper_id}/{note_file})"
        topics = ", ".join((summary.topics if summary else [])[:3]) or "-"
        title = meta.display_title.replace("|", "\\|")[:90]
        lines.append(
            f"| {link} | {meta.kind} | {title} | {'yes' if summary else 'no'} | {topics} |"
        )

    path = store.settings.knowledge_dir / "index.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: object = None) -> object:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default
