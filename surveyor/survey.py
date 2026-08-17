"""Surveys as seeds for the knowledge base.

A survey is worth treating differently from a method paper. Its section
hierarchy *is* a taxonomy of the field, written by people who read everything;
its bibliography is a curated reading list; and where a reference is cited tells
you which branch of the taxonomy it belongs to. All three are recoverable from
the LaTeX without asking a model to guess, so the model's job here is narrowed to
naming and explaining a structure that has already been extracted.

Importing several surveys of the same field is where this pays off: the
references they share are the field's real core, and the places their taxonomies
disagree are the field's live arguments.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .config import Language
from .ingest.arxiv import search_by_title
from .ingest.tex import (
    BibEntry,
    CitationUse,
    Heading,
    parse_bib_entries,
    scan_citations,
    section_tree,
)
from .llm import LLMClient, as_list, as_text
from .models import (
    PaperMeta,
    PaperSummary,
    SectionDigest,
    SurveyNote,
    SurveyReference,
    TaxonomyNode,
)
from .store import PaperStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research advisor mapping out a field from its survey literature. "
    "You are given the survey's real section hierarchy and its real citation "
    "placements, so you never need to invent structure or references: your job is "
    "to name, explain and organize what is already there. You are explicit about "
    "what a survey covers and what it leaves out."
)

_SURVEY_WORDS = re.compile(
    r"\b(survey|review|overview|taxonomy|systematic study|comprehensive study|"
    r"a tutorial|literature review)\b",
    re.IGNORECASE,
)
# Sections that are scaffolding rather than part of the taxonomy. "Summary"
# matters here because surveys close every branch with one, and its citations
# belong to the branch above it.
_NON_TAXONOMY = re.compile(
    r"^(introduction|abstract|background|preliminar|related work|conclusion|"
    r"discussion|acknowledg|references|appendix|notation|organization|"
    r"paper organization|outline|scope|summary|overview|remarks?)",
    re.IGNORECASE,
)


def _taxonomy_path(path: str) -> str:
    """Fold a scaffolding leaf such as "Audio Control > Summary" into its parent."""
    parts = path.split(" > ")
    while len(parts) > 1 and _NON_TAXONOMY.match(parts[-1]):
        parts.pop()
    return " > ".join(parts)


def looks_like_survey(meta: PaperMeta, fulltext: str = "") -> bool:
    """Guess whether a paper is a survey, from its title, comment and abstract."""
    if _SURVEY_WORDS.search(meta.title or ""):
        return True
    if _SURVEY_WORDS.search(meta.comment or ""):
        return True
    opening = (meta.abstract or fulltext[:2000]).lower()
    return bool(
        re.search(r"\b(in|this) (paper|survey|review) (we )?(present|provide|survey)"
                  r"[^.]{0,60}\b(survey|review|overview|taxonomy)\b", opening)
        or re.search(r"\bwe (survey|review)\b", opening)
    )


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def _rank_sections(counts: dict[str, int]) -> list[str]:
    """Order the sections a work is cited in, most telling first.

    A mention in the introduction says nothing about where a work belongs, so
    taxonomy sections outrank scaffolding ones regardless of count.
    """

    folded: dict[str, int] = {}
    for path, count in counts.items():
        collapsed = _taxonomy_path(path)
        folded[collapsed] = folded.get(collapsed, 0) + count

    def rank(path: str) -> tuple[int, int, int]:
        is_taxonomy = 0 if _NON_TAXONOMY.match(path.split(" > ")[-1]) else 1
        return (is_taxonomy, folded[path], path.count(" > "))

    return sorted(folded, key=rank, reverse=True)


def collect_references(store: PaperStore, paper_id: str) -> list[SurveyReference]:
    """Merge a survey's bibliography with where each entry is actually cited.

    Entries never cited in the body are dropped: bibliographies routinely carry
    leftovers, and an uncited entry tells us nothing about the field.
    """
    source = store.source_dir(paper_id)
    if not source.is_dir():
        return []

    entries: dict[str, BibEntry] = parse_bib_entries(source)
    uses: dict[str, CitationUse] = scan_citations(source)
    known = _library_arxiv_ids(store)

    references: list[SurveyReference] = []
    for key, use in uses.items():
        entry = entries.get(key)
        if entry is None:
            continue
        arxiv_id = entry.arxiv_id
        base = arxiv_id.split("v")[0] if arxiv_id else None
        references.append(
            SurveyReference(
                key=key,
                title=entry.title or entry.raw[:160],
                year=entry.year,
                arxiv_id=arxiv_id,
                citations=use.count,
                table_citations=use.table_count,
                sections=_rank_sections(use.sections),
                in_library=bool(base and base in known),
            )
        )

    # Prose citations first: being discussed beats being listed in a table.
    references.sort(key=lambda ref: (-ref.citations, -ref.table_citations, ref.key))
    store.save_references(paper_id, references)
    return references


def _library_arxiv_ids(store: PaperStore) -> set[str]:
    return {
        record.meta.arxiv_id
        for record in store.iter_records()
        if record.meta.arxiv_id
    }


def refresh_reference_status(store: PaperStore) -> None:
    """Re-mark which harvested references are now in the library."""
    known = _library_arxiv_ids(store)
    for paper_id in store.list_surveys():
        references = store.load_references(paper_id)
        if not references:
            continue
        for reference in references:
            base = reference.arxiv_id.split("v")[0] if reference.arxiv_id else None
            reference.in_library = bool(base and base in known)
        store.save_references(paper_id, references)


def resolve_missing_ids(
    store: PaperStore,
    paper_id: str,
    *,
    limit: int = 30,
    progress: Any = None,
) -> int:
    """Look up arXiv ids for the most-cited references that lack one.

    Surveys often cite the conference version of a paper with no arXiv number
    attached, which hides it from harvesting. One title search per reference is
    slow, so only the most-cited ones are worth the round trips.
    """
    references = store.load_references(paper_id) or collect_references(store, paper_id)
    unresolved = [
        reference
        for reference in references
        if not reference.arxiv_id and reference.title
    ][:limit]
    if not unresolved:
        return 0

    known = _library_arxiv_ids(store)
    found = 0
    for reference in unresolved:
        if progress:
            progress(f"looking up: {reference.title[:60]}")
        meta = search_by_title(reference.title, store.settings.arxiv)
        if meta is None or not meta.arxiv_id:
            continue
        reference.arxiv_id = meta.arxiv_id
        reference.title = meta.title or reference.title
        reference.in_library = meta.arxiv_id.split("v")[0] in known
        found += 1

    store.save_references(paper_id, references)
    return found


def reading_list(
    references: list[SurveyReference],
    *,
    limit: int = 20,
    section: str | None = None,
    only_arxiv: bool = True,
    only_missing: bool = False,
) -> list[SurveyReference]:
    """The most-cited references, optionally narrowed to one taxonomy branch."""
    selected = references
    if section:
        needle = section.lower()
        selected = [
            reference
            for reference in selected
            if any(needle in path.lower() for path in reference.sections)
        ]
    if only_arxiv:
        selected = [reference for reference in selected if reference.arxiv_id]
    if only_missing:
        selected = [reference for reference in selected if not reference.in_library]
    return selected[:limit]


def core_references(
    store: PaperStore, survey_ids: list[str], *, min_surveys: int = 2
) -> list[tuple[SurveyReference, int, list[str]]]:
    """References that several surveys agree on, most-agreed first.

    Independent surveys converging on the same work is a much stronger signal
    than citation count inside any one of them, and it needs no model at all.
    Matching is by arXiv id, the only identifier that survives differing citation
    key conventions.
    """
    by_arxiv: dict[str, list[tuple[str, SurveyReference]]] = {}
    for survey_id in survey_ids:
        for reference in store.load_references(survey_id):
            if not reference.arxiv_id:
                continue
            base = reference.arxiv_id.split("v")[0]
            by_arxiv.setdefault(base, []).append((survey_id, reference))

    shared: list[tuple[SurveyReference, int, list[str]]] = []
    for occurrences in by_arxiv.values():
        surveys = sorted({survey_id for survey_id, _ in occurrences})
        if len(surveys) < min_surveys:
            continue
        # Keep whichever record has the most informative title.
        best = max((reference for _, reference in occurrences), key=lambda r: len(r.title))
        best.citations = sum(reference.citations for _, reference in occurrences)
        shared.append((best, len(surveys), surveys))

    shared.sort(key=lambda item: (-item[1], -item[0].citations))
    return shared


# ---------------------------------------------------------------------------
# The survey note
# ---------------------------------------------------------------------------

def _outline_text(headings: list[Heading], limit: int = 200) -> str:
    return "\n".join(
        f"{'  ' * (heading.level - 1)}- {heading.title}" for heading in headings[:limit]
    )


def _taxonomy_evidence(
    headings: list[Heading], references: list[SurveyReference], per_section: int = 8
) -> str:
    """For each candidate taxonomy section, list the works cited under it."""
    by_section: dict[str, list[SurveyReference]] = {}
    for reference in references:
        for path in reference.sections:
            by_section.setdefault(_taxonomy_path(path), []).append(reference)

    interesting = [
        heading.title
        for heading in headings
        if heading.level <= 3 and not _NON_TAXONOMY.match(heading.title)
    ]
    blocks: list[str] = []
    for path, refs in by_section.items():
        leaf = path.split(" > ")[-1]
        if leaf not in interesting:
            continue
        top = sorted(refs, key=lambda ref: (-ref.citations, -ref.table_citations))[
            :per_section
        ]
        listed = "; ".join(
            f"{ref.title[:70] or ref.key}" + (f" [{ref.arxiv_id}]" if ref.arxiv_id else "")
            for ref in top
        )
        blocks.append(f"- {path}\n    cited here: {listed}")
    return "\n".join(blocks)


def _parse_taxonomy(raw: Any, depth: int = 0) -> list[TaxonomyNode]:
    if depth > 3 or not isinstance(raw, list):
        return []
    nodes: list[TaxonomyNode] = []
    for item in raw:
        if isinstance(item, str):
            nodes.append(TaxonomyNode(name=item.strip()))
            continue
        if not isinstance(item, dict):
            continue
        name = as_text(item.get("name") or item.get("category") or item.get("title"))
        if not name:
            continue
        nodes.append(
            TaxonomyNode(
                name=name,
                description=as_text(item.get("description")),
                representative=as_list(item.get("representative"), limit=12),
                children=_parse_taxonomy(item.get("children"), depth + 1),
            )
        )
    return nodes


_SCHEMA = """Return a single JSON object with exactly these keys:

{
  "field_name": "the canonical name of the field this survey covers, English, "
                "Title Case, e.g. 'Controllable Video Generation'",
  "scope": "what the survey covers, what it deliberately excludes, and roughly "
           "what time range of work it spans. 3-5 sentences.",
  "taxonomy": [
    {
      "name": "branch name",
      "description": "what defines this branch and what distinguishes it from "
                     "its siblings, 2-4 sentences",
      "representative": ["arXiv ids or short titles the survey files here"],
      "children": [ {"name": "...", "description": "...", "representative": []} ]
    }
  ],
  "key_concepts": ["terms you must know to read this field"],
  "benchmarks": ["datasets, benchmarks and metrics the field evaluates on"],
  "milestones": ["landmark works in rough chronological order, with why each mattered"],
  "consensus": ["what the survey presents as settled"],
  "open_challenges": ["what the survey names as unsolved"],
  "reading_path": ["5-8 steps for someone entering the field, each with a reason"]
}

Rules:
- Build "taxonomy" from the survey's own section hierarchy given below. Merge or
  rename sections only when the heading is uninformative, and drop scaffolding
  sections such as Introduction, Related Work and Conclusion.
- Put entries in "representative" only if they appear in the citation evidence.
- Output raw JSON only, no Markdown fence, no commentary."""


def summarize_survey(
    store: PaperStore,
    paper_id: str,
    *,
    language: Language | None = None,
    client: LLMClient | None = None,
) -> SurveyNote:
    """Build and persist a survey's note, taxonomy and reference list."""
    settings = store.settings
    language = language or settings.default_language
    client = client or LLMClient(settings.llm)

    meta = store.load_meta(paper_id)
    if meta is None:
        raise ValueError(f"unknown paper: {paper_id}")

    fulltext = store.load_fulltext(paper_id) or store.rebuild_fulltext(paper_id)
    source = store.source_dir(paper_id)
    headings = section_tree(source) if source.is_dir() else []
    references = collect_references(store, paper_id)

    budget = int(settings.llm.max_context_chars * 0.7)
    outline = _outline_text(headings)
    evidence = _taxonomy_evidence(headings, references)

    # The outline and citation evidence are compact and always sent. The body is
    # only useful for the branch descriptions, so it yields space first.
    scaffolding = len(outline) + len(evidence) + len(_SCHEMA) + 4000
    body = fulltext[: max(budget - scaffolding, 0)]
    digests: list[SectionDigest] = []
    if len(fulltext) > budget - scaffolding:
        digests = _digest_survey_sections(client, meta, store, language, budget // 3)
        if digests:
            body = "\n\n".join(
                f"## {digest.section}\n{digest.summary}" for digest in digests
            )

    prompt = (
        f"Survey: {meta.display_title}\n"
        f"Authors: {meta.author_line}\n"
        f"Abstract: {meta.abstract}\n\n"
        f"--- SECTION HIERARCHY (the survey's own structure) ---\n{outline}\n\n"
        f"--- CITATION EVIDENCE (which works are cited under which section) ---\n"
        f"{evidence or '(no usable bibliography found)'}\n\n"
        f"--- BODY ---\n{body}\n--- END BODY ---\n\n"
        f"{_SCHEMA}\n\n{language.instruction}\n"
        "The JSON keys stay in English exactly as specified; only values follow "
        "the language rule. Keep arXiv ids and English technical terms unchanged."
    )

    data = client.complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    note = SurveyNote(
        paper_id=paper_id,
        language=language.value,
        model=settings.llm.model,
        field_name=as_text(data.get("field_name")) or meta.display_title,
        scope=as_text(data.get("scope")),
        taxonomy=_parse_taxonomy(data.get("taxonomy")),
        key_concepts=as_list(data.get("key_concepts"), limit=40),
        benchmarks=as_list(data.get("benchmarks"), limit=30),
        milestones=as_list(data.get("milestones"), limit=30),
        consensus=as_list(data.get("consensus")),
        open_challenges=as_list(data.get("open_challenges")),
        reading_path=as_list(data.get("reading_path"), limit=12),
        section_digests=digests,
    )

    meta.kind = "survey"
    store.save_meta(meta)
    store.save_survey(note, render_survey_md(meta, note, references))
    # A derived summary keeps surveys visible to list, search and retrieval.
    store.save_summary(
        _derived_summary(note), _derived_summary_markdown(meta, note)
    )
    return note


def _digest_survey_sections(
    client: LLMClient,
    meta: PaperMeta,
    store: PaperStore,
    language: Language,
    window: int,
) -> list[SectionDigest]:
    """Condense a long survey section by section so the taxonomy pass fits."""
    windows: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for chunk in store.chunks(meta.paper_id, target_chars=6000):
        if size + len(chunk.text) > window and current:
            windows.append(current)
            current, size = [], 0
        current.append((chunk.section, chunk.text))
        size += len(chunk.text)
    if current:
        windows.append(current)

    digests: list[SectionDigest] = []
    for index, group in enumerate(windows, start=1):
        body = "\n\n".join(f"## {section}\n{text}" for section, text in group)
        prompt = (
            f"Survey: {meta.display_title}\n"
            f"Part {index} of {len(windows)}.\n\n{body}\n\n"
            'Return JSON: {"sections": [{"section": "...", "summary": "3-5 sentences '
            'covering which approaches this section groups together and how it '
            'distinguishes them", "key_points": ["..."]}]}\n'
            "Name concrete methods where the text does. Output raw JSON only.\n\n"
            f"{language.instruction}"
        )
        try:
            data = client.complete_json(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tier="fast",
            )
        except Exception as exc:
            log.warning("%s: survey digest %d failed: %s", meta.paper_id, index, exc)
            continue
        digests.extend(
            SectionDigest(
                section=as_text(item.get("section")) or "Untitled",
                summary=as_text(item.get("summary")),
                key_points=as_list(item.get("key_points"), limit=8),
            )
            for item in data.get("sections") or []
            if isinstance(item, dict)
        )
    return digests


def _derived_summary(note: SurveyNote) -> PaperSummary:
    """A stand-in note so a survey still shows up in the ordinary views."""
    branches = [name for _depth, node in note.flat_taxonomy() for name in (node.name,)]
    return PaperSummary(
        paper_id=note.paper_id,
        language=note.language,
        model=note.model,
        one_liner=f"Survey of {note.field_name}." if note.field_name else "Survey.",
        problem=note.scope,
        method="Survey: organizes the field as " + "; ".join(branches[:12])
        if branches
        else "Survey.",
        contributions=note.consensus[:6],
        results=note.milestones[:8],
        limitations=note.open_challenges[:8],
        concepts=note.key_concepts,
        topics=[note.field_name] if note.field_name else [],
        datasets=note.benchmarks,
        open_questions=note.open_challenges,
    )


def _derived_summary_markdown(meta: PaperMeta, note: SurveyNote) -> str:
    return (
        f"# {meta.display_title}\n\n"
        f"This is a survey. See [survey.md](survey.md) for its taxonomy of "
        f"{note.field_name or 'the field'} and its reading list.\n\n"
        f"{note.scope}\n"
    )


def render_survey_md(
    meta: PaperMeta, note: SurveyNote, references: list[SurveyReference]
) -> str:
    """Render a survey note, taxonomy tree and prioritized reading list."""
    lines: list[str] = [f"# {meta.display_title}", ""]

    facts = [f"**{meta.author_line}**"]
    if meta.abs_url:
        facts.append(f"[arXiv:{meta.arxiv_id}]({meta.abs_url})")
    if meta.published:
        facts.append(meta.published[:10])
    lines += [" · ".join(facts), ""]

    if note.field_name:
        lines += [f"**Field:** {note.field_name}", ""]
    if note.scope:
        lines += ["## Scope", "", note.scope, ""]

    if note.taxonomy:
        lines += ["## Taxonomy", ""]
        for depth, node in note.flat_taxonomy():
            indent = "  " * (depth - 1)
            lines.append(f"{indent}- **{node.name}**")
            if node.description:
                lines.append(f"{indent}  {node.description}")
            if node.representative:
                lines.append(f"{indent}  *Representative:* {', '.join(node.representative)}")
        lines.append("")

    def bullets(title: str, items: list[str]) -> None:
        if items:
            lines.extend([f"## {title}", "", *[f"- {item}" for item in items], ""])

    bullets("Key concepts", note.key_concepts)
    bullets("Benchmarks and metrics", note.benchmarks)
    bullets("Milestones", note.milestones)
    bullets("Consensus", note.consensus)
    bullets("Open challenges", note.open_challenges)
    bullets("Suggested reading path", note.reading_path)

    top = reading_list(references, limit=30, only_arxiv=True)
    if top:
        lines += [
            "## Most-cited references with arXiv sources",
            "",
            (
                "Ranked by how often this survey cites them in prose. "
                "`paper survey harvest <id>` ingests them."
            ),
            "",
            "| Cited | arXiv | In library | Reference |",
            "| --- | --- | --- | --- |",
        ]
        for reference in top:
            title = (reference.title or reference.key).replace("|", "\\|")[:80]
            lines.append(
                f"| {reference.citations} | `{reference.arxiv_id}` | "
                f"{'yes' if reference.in_library else '-'} | {title} |"
            )
        lines.append("")

    total = len(references)
    with_id = sum(1 for reference in references if reference.arxiv_id)
    lines += [
        "---",
        "",
        (
            f"*{total} cited references, {with_id} with arXiv sources. "
            f"Generated by surveyor using {note.model} on {note.generated_at[:10]}.*"
        ),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-survey synthesis
# ---------------------------------------------------------------------------

def group_surveys_by_field(store: PaperStore) -> dict[str, list[str]]:
    """Group survey ids by the field name their notes claim."""
    grouped: dict[str, list[str]] = {}
    for paper_id in store.list_surveys():
        note = store.load_survey(paper_id)
        field_name = (note.field_name if note else "") or "Unclassified"
        grouped.setdefault(field_name, []).append(paper_id)
    return dict(sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])))


def _survey_block(store: PaperStore, paper_id: str) -> str:
    meta = store.load_meta(paper_id)
    note = store.load_survey(paper_id)
    if meta is None or note is None:
        return ""
    outline = "\n".join(
        f"{'  ' * (depth - 1)}- {node.name}: {node.description[:200]}"
        for depth, node in note.flat_taxonomy()
    )
    return "\n".join(
        [
            f"### [{paper_id}] {meta.display_title}",
            f"Published: {(meta.published or '?')[:10]}",
            f"Field as stated: {note.field_name}",
            f"Scope: {note.scope}",
            "Taxonomy:",
            outline,
            f"Consensus: {'; '.join(note.consensus)}",
            f"Open challenges: {'; '.join(note.open_challenges)}",
            f"Benchmarks: {', '.join(note.benchmarks)}",
        ]
    )


def merge_surveys(
    store: PaperStore,
    survey_ids: list[str],
    *,
    field_name: str = "",
    language: Language | None = None,
    client: LLMClient | None = None,
    collection: str | None = None,
) -> Path:
    """Reconcile several surveys of one field into a single map.

    Writes ``fields/<slug>.md`` under the knowledge folder of the collection.
    The shared-reference table is computed from the bibliographies rather than
    generated, so the "core reading" list is a fact about the surveys, not an
    opinion of the model.
    """
    from .paths import knowledge_root, note_link, slugify

    settings = store.settings
    language = language or settings.default_language
    client = client or LLMClient(settings.llm)

    blocks = [block for block in (_survey_block(store, sid) for sid in survey_ids) if block]
    if not blocks:
        raise ValueError("none of those surveys have notes yet")

    field_name = field_name or _dominant_field(store, survey_ids)
    shared = core_references(store, survey_ids, min_surveys=2) if len(survey_ids) > 1 else []
    shared_text = "\n".join(
        f"- {reference.display} [{reference.arxiv_id}] — cited by {count} surveys"
        for reference, count, _surveys in shared[:40]
    )

    if len(survey_ids) == 1:
        instruction = (
            "Write a field map based on this single survey: the structure of the "
            "field, what the survey establishes, and what it leaves open. Flag "
            "clearly that this rests on one survey's framing."
        )
    else:
        instruction = (
            "Reconcile these surveys into one map of the field, with sections:\n"
            "1. **The field** — what it is, in plain language.\n"
            "2. **Merged taxonomy** — one structure that accommodates all of them. "
            "Where they carve the field the same way, say so; where they differ, "
            "present the alternatives and say which survey proposes which.\n"
            "3. **Where they disagree** — genuine differences in framing, scope or "
            "what counts as solved. Ignore mere wording differences.\n"
            "4. **How the framing changed over time** — compare older to newer "
            "surveys and say what shifted.\n"
            "5. **Coverage gaps** — subareas one survey treats that others omit, "
            "and anything none of them cover.\n"
            "6. **What to read first** — grounded in the shared references below.\n"
            "7. **Open problems** — where the surveys agree work is needed."
        )

    prompt = (
        f"Field: {field_name}\n"
        f"{len(blocks)} survey(s) in the library.\n\n"
        + "\n\n".join(blocks)
        + (
            f"\n\n--- REFERENCES CITED BY MORE THAN ONE SURVEY ---\n{shared_text}\n"
            if shared_text
            else ""
        )
        + f"\n{instruction}\n"
        "Cite surveys as [paper_id]. Be concrete; do not restate the taxonomies "
        "verbatim.\n\n"
        f"{language.instruction}"
    )

    body = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    path = knowledge_root(settings, collection) / "fields" / f"{slugify(field_name)}.md"
    lines = [f"# {field_name}", "", f"Synthesized from {len(blocks)} survey(s).", "", body, ""]
    lines += ["## Surveys", ""]
    for paper_id in survey_ids:
        meta = store.load_meta(paper_id)
        if meta:
            date = (meta.published or "")[:10]
            target = note_link(settings, paper_id, "survey.md", path.parent)
            lines.append(f"- [{paper_id}]({target}) {date} — {meta.display_title}")
    lines.append("")

    if shared:
        lines += [
            "## Core reading: references shared across surveys",
            "",
            (
                "Counted from the surveys' own bibliographies. Independent surveys "
                "converging on the same work is a stronger signal than any single "
                "survey's citation count."
            ),
            "",
            "| Surveys | Cites | arXiv | In library | Reference |",
            "| --- | --- | --- | --- | --- |",
        ]
        for reference, count, _surveys in shared[:40]:
            title = (reference.title or reference.key).replace("|", "\\|")[:70]
            lines.append(
                f"| {count} | {reference.citations} | `{reference.arxiv_id}` | "
                f"{'yes' if reference.in_library else '-'} | {title} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _dominant_field(store: PaperStore, survey_ids: Iterable[str]) -> str:
    counter: Counter[str] = Counter()
    for paper_id in survey_ids:
        note = store.load_survey(paper_id)
        if note and note.field_name:
            counter[note.field_name] += 1
    return counter.most_common(1)[0][0] if counter else "Unclassified"
