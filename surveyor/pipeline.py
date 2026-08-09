"""End-to-end ingestion: reference in, summarized paper on disk out."""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .config import Language
from .ingest import arxiv
from .ingest.resolve import ArxivRef, resolve_inputs
from .ingest.tex import build_fulltext, find_root_tex
from .llm import LLMClient
from .models import PaperMeta
from .store import PaperStore, write_index
from .summarize import summarize_paper
from .survey import (
    collect_references,
    looks_like_survey,
    reading_list,
    refresh_reference_status,
    summarize_survey,
)

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class IngestResult:
    paper_id: str
    title: str = ""
    downloaded: bool = False
    converted: bool = False
    summarized: bool = False
    skipped: bool = False
    is_survey: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def status(self) -> str:
        if self.errors:
            return "failed"
        if self.skipped:
            return "skipped"
        if self.summarized:
            return "survey" if self.is_survey else "summarized"
        if self.converted:
            return "survey refs" if self.is_survey else "ingested"
        return "partial"


def _noop(_message: str) -> None:
    pass


def _title_from_fulltext(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def ingest_ref(
    store: PaperStore,
    ref: ArxivRef,
    *,
    summarize: bool = True,
    force: bool = False,
    language: Language | None = None,
    client: LLMClient | None = None,
    kind: str | None = None,
    progress: Progress = _noop,
) -> IngestResult:
    """Download, convert and summarize one arXiv paper."""
    settings = store.settings

    # Without a pinned version we cannot know the directory name until arXiv
    # tells us which version is current.
    meta = store.load_meta(ref.dirname) if ref.version else None
    if meta is None:
        existing = [
            record.meta
            for record in store.iter_records()
            if record.meta.arxiv_id == ref.base_id
        ]
        meta = existing[0] if existing and not force else None

    if meta is not None and not force:
        result = IngestResult(paper_id=meta.paper_id, title=meta.title, skipped=True)
        if summarize and store.load_summary(meta.paper_id) is None:
            result.skipped = False
            return _summarize_step(store, result, language, client, progress, kind)
        progress(f"{meta.paper_id}: already in the library")
        return result

    progress(f"{ref.canonical}: fetching metadata")
    try:
        fetched = arxiv.fetch_metadata([ref], settings.arxiv)
    except Exception as exc:
        log.warning("metadata lookup failed for %s: %s", ref.canonical, exc)
        fetched = {}

    meta = fetched.get(ref.base_id)
    if meta is None:
        # arXiv may be unreachable or the id may be wrong; keep going with a stub
        # so a manually placed source tree can still be processed.
        meta = PaperMeta(
            paper_id=ref.dirname,
            arxiv_id=ref.base_id,
            version=ref.version,
            abs_url=f"https://arxiv.org/abs/{ref.canonical}",
        )

    result = IngestResult(paper_id=meta.paper_id, title=meta.title)
    source_dir = store.source_dir(meta.paper_id)

    if force and source_dir.is_dir():
        shutil.rmtree(source_dir)

    if not source_dir.is_dir() or not any(source_dir.iterdir()):
        progress(f"{meta.paper_id}: downloading source")
        try:
            kind = arxiv.fetch_and_extract(ref, source_dir, settings.arxiv)
            result.downloaded = True
            if kind == "pdf":
                result.errors.append(
                    "arXiv only has a PDF for this paper; no LaTeX source to read"
                )
        except Exception as exc:
            result.errors.append(f"download failed: {exc}")
            log.warning("%s: download failed: %s", meta.paper_id, exc)
    else:
        result.downloaded = True

    meta.has_source = source_dir.is_dir() and find_root_tex(source_dir) is not None
    store.save_meta(meta)

    if meta.has_source:
        progress(f"{meta.paper_id}: converting LaTeX")
        text = store.rebuild_fulltext(meta.paper_id)
        result.converted = bool(text)
        if text and not meta.title:
            meta.title = _title_from_fulltext(text)
            store.save_meta(meta)
            result.title = meta.title

    result.is_survey = classify_survey(store, meta.paper_id, kind, progress)

    if summarize and (result.converted or meta.abstract):
        return _summarize_step(store, result, language, client, progress, kind)
    return result


def _is_survey(store: PaperStore, paper_id: str, kind: str | None) -> bool:
    """Honour an explicit ``kind``, otherwise sniff the title and abstract."""
    if kind in ("survey", "paper"):
        return kind == "survey"
    meta = store.load_meta(paper_id)
    if meta is None:
        return False
    if meta.kind == "survey":
        return True
    return looks_like_survey(meta, store.load_fulltext(paper_id))


def classify_survey(
    store: PaperStore, paper_id: str, kind: str | None, progress: Progress = _noop
) -> bool:
    """Mark a survey and extract its bibliography. Returns whether it is one.

    Kept separate from summarization because both the citation graph and the
    section hierarchy come straight out of the LaTeX: they cost no tokens and are
    worth having even when no model is configured.
    """
    if not _is_survey(store, paper_id, kind):
        return False

    meta = store.load_meta(paper_id)
    if meta is not None and meta.kind != "survey":
        meta.kind = "survey"
        store.save_meta(meta)

    progress(f"{paper_id}: extracting bibliography")
    references = collect_references(store, paper_id)
    on_arxiv = sum(1 for reference in references if reference.arxiv_id)
    progress(
        f"{paper_id}: {len(references)} cited references, {on_arxiv} with arXiv sources"
    )
    return True


def _summarize_step(
    store: PaperStore,
    result: IngestResult,
    language: Language | None,
    client: LLMClient | None,
    progress: Progress,
    kind: str | None = None,
) -> IngestResult:
    if _is_survey(store, result.paper_id, kind):
        result.is_survey = True
        progress(f"{result.paper_id}: reading as a survey (taxonomy + references)")
        try:
            note = summarize_survey(
                store, result.paper_id, language=language, client=client
            )
            result.summarized = True
            result.title = result.title or f"Survey: {note.field_name}"
        except Exception as exc:
            result.errors.append(f"survey analysis failed: {exc}")
            log.warning("%s: survey analysis failed: %s", result.paper_id, exc)
        return result

    progress(f"{result.paper_id}: summarizing")
    try:
        summary = summarize_paper(
            store, result.paper_id, language=language, client=client
        )
        result.summarized = True
        result.title = result.title or summary.one_liner[:80]
    except Exception as exc:
        result.errors.append(f"summarization failed: {exc}")
        log.warning("%s: summarization failed: %s", result.paper_id, exc)
    return result


def ingest_inputs(
    store: PaperStore,
    inputs: Sequence[str],
    *,
    summarize: bool = True,
    force: bool = False,
    language: Language | None = None,
    kind: str | None = None,
    progress: Progress = _noop,
) -> list[IngestResult]:
    """Ingest everything referenced by ``inputs`` (ids, URLs, list files, prose)."""
    refs = resolve_inputs(inputs)
    if not refs:
        return []

    client = LLMClient(store.settings.llm) if summarize else None
    results = [
        ingest_ref(
            store,
            ref,
            summarize=summarize,
            force=force,
            language=language,
            client=client,
            kind=kind,
            progress=progress,
        )
        for ref in refs
    ]
    write_index(store)
    # New papers may satisfy references that surveys were still missing.
    refresh_reference_status(store)
    return results


# ---------------------------------------------------------------------------
# Adopting source trees that are already on disk
# ---------------------------------------------------------------------------

_DIRNAME_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")


def adopt_directory(
    store: PaperStore,
    directory: Path,
    *,
    move: bool = True,
    summarize: bool = True,
    language: Language | None = None,
    kind: str | None = None,
    progress: Progress = _noop,
) -> IngestResult:
    """Bring an already-extracted source tree into the library.

    Used to migrate folders like ``arXiv-2503.07598v2/`` that were unpacked by
    hand before the agent existed.
    """
    directory = directory.resolve()
    match = _DIRNAME_ID.search(directory.name)
    ref = (
        ArxivRef(base_id=match.group(1), version=match.group(2)) if match else None
    )
    paper_id = ref.dirname if ref else _slugify(directory.name)

    if store.exists(paper_id):
        progress(f"{paper_id}: already in the library")
        return IngestResult(paper_id=paper_id, skipped=True)

    meta: PaperMeta | None = None
    if ref is not None:
        progress(f"{paper_id}: fetching metadata")
        try:
            meta = arxiv.fetch_metadata([ref], store.settings.arxiv).get(ref.base_id)
        except Exception as exc:
            log.warning("metadata lookup failed for %s: %s", ref.canonical, exc)
    if meta is None:
        meta = PaperMeta(
            paper_id=paper_id,
            arxiv_id=ref.base_id if ref else None,
            version=ref.version if ref else None,
            abs_url=f"https://arxiv.org/abs/{ref.canonical}" if ref else None,
            source_kind="arxiv" if ref else "local",
        )
    # arXiv reports the current version; keep the directory the user already has.
    meta.paper_id = paper_id

    destination = store.source_dir(paper_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    progress(f"{paper_id}: {'moving' if move else 'copying'} source")
    if move:
        shutil.move(str(directory), str(destination))
    else:
        shutil.copytree(directory, destination)

    result = IngestResult(paper_id=paper_id, title=meta.title, downloaded=True)
    meta.has_source = find_root_tex(destination) is not None
    store.save_meta(meta)

    if meta.has_source:
        progress(f"{paper_id}: converting LaTeX")
        text = store.rebuild_fulltext(paper_id)
        result.converted = bool(text)
        if text and not meta.title:
            meta.title = _title_from_fulltext(text)
            store.save_meta(meta)
        result.title = meta.title
    else:
        result.errors.append("no LaTeX root file found in this directory")

    result.is_survey = classify_survey(store, paper_id, kind, progress)

    if summarize and result.converted:
        _summarize_step(store, result, language, None, progress, kind)
    return result


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return slug or "paper"


def harvest_survey(
    store: PaperStore,
    survey_id: str,
    *,
    limit: int = 20,
    section: str | None = None,
    only_missing: bool = True,
    summarize: bool = True,
    language: Language | None = None,
    progress: Progress = _noop,
) -> list[IngestResult]:
    """Ingest the papers a survey cites most, turning it into a reading list."""
    references = store.load_references(survey_id) or collect_references(store, survey_id)
    picks = reading_list(
        references,
        limit=limit,
        section=section,
        only_arxiv=True,
        only_missing=only_missing,
    )
    if not picks:
        return []
    progress(f"{survey_id}: harvesting {len(picks)} referenced paper(s)")
    return ingest_inputs(
        store,
        [reference.arxiv_id for reference in picks if reference.arxiv_id],
        summarize=summarize,
        language=language,
        kind="paper",
        progress=progress,
    )


def rebuild_all(store: PaperStore, progress: Progress = _noop) -> int:
    """Re-run LaTeX conversion for every paper, e.g. after a parser improvement."""
    count = 0
    for paper_id in store.list_ids():
        if not store.source_dir(paper_id).is_dir():
            continue
        progress(f"{paper_id}: reconverting")
        if store.rebuild_fulltext(paper_id):
            count += 1
    write_index(store)
    return count


def source_stats(store: PaperStore, paper_id: str) -> dict[str, object]:
    """Quick facts about one paper's source tree, for diagnostics."""
    source = store.source_dir(paper_id)
    root = find_root_tex(source) if source.is_dir() else None
    text, _ = build_fulltext(source) if source.is_dir() else ("", None)
    return {
        "paper_id": paper_id,
        "root_tex": str(root.relative_to(source)) if root else None,
        "tex_files": len(list(source.rglob("*.tex"))) if source.is_dir() else 0,
        "fulltext_chars": len(text),
    }
