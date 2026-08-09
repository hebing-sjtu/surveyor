"""Cross-paper synthesis: topic digests, a concept glossary, a library overview.

Individual notes answer "what does this paper say". These answer "what does my
reading add up to", which is the part that usually never gets written down.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import Language
from .llm import LLMClient
from .models import PaperRecord
from .store import PaperStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a research advisor synthesizing a reading list into durable knowledge. "
    "You draw connections the individual papers do not state themselves, but you never "
    "invent results. Every claim about a specific paper carries its id in brackets, "
    "like [2503.07598v2]. You are candid about disagreements and weak evidence."
)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "misc"


def topic_index(store: PaperStore) -> dict[str, list[PaperRecord]]:
    """Group summarized papers by topic label."""
    grouped: dict[str, list[PaperRecord]] = defaultdict(list)
    for record in store.iter_records():
        if not record.summary:
            continue
        for topic in record.summary.topics or ["Uncategorized"]:
            grouped[topic].append(record)
    return dict(sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])))


def _record_block(record: PaperRecord) -> str:
    meta, summary = record.meta, record.summary
    if summary is None:
        return f"### [{meta.paper_id}] {meta.display_title}\n{meta.abstract}"
    parts = [
        f"### [{meta.paper_id}] {meta.display_title}",
        f"Published: {(meta.published or '?')[:10]}",
        f"One-liner: {summary.one_liner}",
        f"Problem: {summary.problem}",
        f"Method: {summary.method}",
    ]
    if summary.contributions:
        parts.append("Contributions: " + "; ".join(summary.contributions))
    if summary.results:
        parts.append("Results: " + "; ".join(summary.results))
    if summary.limitations:
        parts.append("Limitations: " + "; ".join(summary.limitations))
    if summary.datasets:
        parts.append("Datasets: " + ", ".join(summary.datasets))
    if summary.baselines:
        parts.append("Baselines: " + ", ".join(summary.baselines))
    if summary.open_questions:
        parts.append("Open questions: " + "; ".join(summary.open_questions))
    return "\n".join(parts)


def write_topic_digest(
    store: PaperStore,
    topic: str,
    records: list[PaperRecord],
    *,
    language: Language | None = None,
    client: LLMClient | None = None,
) -> Path:
    """Synthesize one topic into ``knowledge/topics/<slug>.md``."""
    settings = store.settings
    language = language or settings.default_language
    client = client or LLMClient(settings.llm)

    ordered = sorted(records, key=lambda record: record.meta.published or "")
    budget = int(settings.llm.max_context_chars * 0.7)

    blocks: list[str] = []
    used = 0
    for record in ordered:
        block = _record_block(record)
        if used + len(block) > budget:
            break
        blocks.append(block)
        used += len(block)

    prompt = (
        f"Research area: {topic}\n"
        f"Papers in the library for this area ({len(ordered)}), oldest first:\n\n"
        + "\n\n".join(blocks)
        + "\n\nWrite a synthesis of this area with these sections:\n"
        "1. **What this area is about** — the shared problem, in plain language.\n"
        "2. **Lines of work** — group the papers by the approach they take, not one "
        "paragraph per paper. Name each line and say which papers belong to it.\n"
        "3. **How the ideas build on each other** — the trajectory over time, "
        "including where later work contradicts or supersedes earlier work.\n"
        "4. **Consensus** — what these papers agree on.\n"
        "5. **Open disagreements and gaps** — where they conflict, what evidence is "
        "missing, which comparisons nobody has run.\n"
        "6. **Shared vocabulary** — the terms you need to read this area.\n"
        "7. **Suggested reading order** — for someone starting today, with a reason "
        "for each step.\n\n"
        "Cite papers as [paper_id]. Prefer specifics over generalities.\n\n"
        f"{language.instruction}"
    )

    body = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    lines = [f"# {topic}", "", f"{len(ordered)} papers in the library.", "", body, "",
             "## Papers", ""]
    for record in ordered:
        meta = record.meta
        date = (meta.published or "")[:10]
        lines.append(
            f"- [{meta.paper_id}](../../papers/{meta.paper_id}/summary.md) "
            f"{date} — {meta.display_title}"
        )

    path = settings.knowledge_dir / "topics" / f"{slugify(topic)}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_all_topic_digests(
    store: PaperStore,
    *,
    min_papers: int = 2,
    language: Language | None = None,
    client: LLMClient | None = None,
    only: str | None = None,
) -> list[Path]:
    """Regenerate topic digests for every topic with enough papers behind it."""
    client = client or LLMClient(store.settings.llm)
    written: list[Path] = []
    for topic, records in topic_index(store).items():
        if only and slugify(only) != slugify(topic):
            continue
        if len(records) < min_papers and not only:
            continue
        try:
            written.append(
                write_topic_digest(
                    store, topic, records, language=language, client=client
                )
            )
        except Exception as exc:
            log.warning("topic digest failed for %s: %s", topic, exc)
    return written


def write_glossary(store: PaperStore) -> Path:
    """Concept -> papers, built directly from the notes without calling a model."""
    concepts: dict[str, list[str]] = defaultdict(list)
    for record in store.iter_records():
        if not record.summary:
            continue
        for concept in record.summary.concepts:
            key = concept.strip()
            if key and record.meta.paper_id not in concepts[key]:
                concepts[key].append(record.meta.paper_id)

    lines = ["# Concept glossary", "",
             "Concepts extracted from paper notes, with the papers that use them.", ""]
    for concept in sorted(concepts, key=lambda name: (-len(concepts[name]), name.lower())):
        papers = ", ".join(
            f"[{pid}](../papers/{pid}/summary.md)" for pid in concepts[concept]
        )
        lines.append(f"- **{concept}** — {papers}")

    path = store.settings.knowledge_dir / "glossary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_overview(
    store: PaperStore,
    *,
    language: Language | None = None,
    client: LLMClient | None = None,
) -> Path:
    """A single page describing what the whole library is about."""
    settings = store.settings
    language = language or settings.default_language
    client = client or LLMClient(settings.llm)

    records = [record for record in store.iter_records() if record.summary]
    if not records:
        raise ValueError("no summarized papers yet")

    topics = Counter(
        topic for record in records for topic in (record.summary.topics if record.summary else [])
    )
    lines = [
        f"- [{record.meta.paper_id}] {record.meta.display_title} — "
        f"{record.summary.one_liner} (topics: {', '.join(record.summary.topics)})"
        for record in records
        if record.summary
    ]

    prompt = (
        f"A researcher's library of {len(records)} papers.\n"
        f"Topic distribution: {', '.join(f'{t} ({c})' for t, c in topics.most_common())}\n\n"
        + "\n".join(lines)
        + "\n\nWrite a one-page overview: what this researcher is working on, the "
        "clusters their reading falls into and how those clusters relate, the "
        "through-lines that connect otherwise separate papers, and the obvious gaps "
        "worth reading next. Cite papers as [paper_id]. Be specific and avoid "
        "restating the list.\n\n"
        f"{language.instruction}"
    )
    body = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )

    path = settings.knowledge_dir / "overview.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# Library overview\n\n{body}\n", encoding="utf-8")
    return path
