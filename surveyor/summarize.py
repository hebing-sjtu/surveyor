"""Turn a paper's full text into a structured, reusable note."""

from __future__ import annotations

import logging
from collections import Counter

from .config import Language
from .llm import LLMClient, as_list, as_text
from .models import PaperMeta, PaperSummary, SectionDigest
from .store import PaperStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a meticulous research assistant helping a machine-learning researcher "
    "build a long-term reading library. You read papers closely and report what they "
    "actually claim, never inventing numbers, datasets or citations. When the source "
    "text is ambiguous or a detail is missing, say so instead of guessing."
)

_SCHEMA = """Return a single JSON object with exactly these keys:

{
  "one_liner": "one sentence: what this paper does and why it matters",
  "problem": "the concrete problem being solved, 2-4 sentences",
  "motivation": "why existing work is insufficient, 2-4 sentences",
  "method": "how the method works, in enough technical detail that the reader could "
            "explain it at a reading group: inputs, architecture, training objective, "
            "key design decisions and the reasoning behind them. 200-400 words.",
  "contributions": ["each distinct claimed contribution"],
  "experiments": "setup: tasks, datasets, baselines, metrics, ablations. 100-200 words.",
  "results": ["concrete findings, with the actual numbers when the paper gives them"],
  "limitations": ["limitations the paper admits, plus ones you can see yourself"],
  "concepts": ["technical concepts worth remembering, as short English noun phrases"],
  "topics": ["2-4 coarse research areas, English, Title Case"],
  "datasets": ["datasets and benchmarks used"],
  "baselines": ["methods compared against"],
  "open_questions": ["questions this paper leaves open or provokes"],
  "critique": "your honest assessment: how strong is the evidence, what would you "
              "want to check, what is over-claimed. 80-150 words.",
  "section_digests": [
    {"section": "section heading", "summary": "2-3 sentences", "key_points": ["..."]}
  ]
}

Rules:
- Output raw JSON only, no Markdown fence, no commentary.
- Every string value must be plain text; no nested objects beyond the shape above.
- "topics" is used to group papers across the library, so prefer established area
  names over paper-specific phrasing."""


def _language_block(language: Language) -> str:
    return (
        f"{language.instruction}\n"
        "The JSON keys themselves stay in English exactly as specified; only the "
        "values follow the language rule above."
    )


def _known_topics(store: PaperStore, limit: int = 40) -> list[str]:
    counter: Counter[str] = Counter()
    for record in store.iter_records():
        if record.summary:
            counter.update(record.summary.topics)
    return [topic for topic, _count in counter.most_common(limit)]


def _paper_header(meta: PaperMeta) -> str:
    lines = [f"Title: {meta.display_title}"]
    if meta.authors:
        lines.append(f"Authors: {meta.author_line}")
    if meta.categories:
        lines.append(f"arXiv categories: {', '.join(meta.categories)}")
    if meta.published:
        lines.append(f"Published: {meta.published[:10]}")
    if meta.comment:
        lines.append(f"Author comment: {meta.comment}")
    if meta.abstract:
        lines.append(f"\nAbstract:\n{meta.abstract}")
    return "\n".join(lines)


def _digest_sections(
    client: LLMClient,
    meta: PaperMeta,
    store: PaperStore,
    language: Language,
    budget: int,
) -> list[SectionDigest]:
    """Map pass: summarize the paper one window of sections at a time."""
    windows: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    size = 0
    for chunk in store.chunks(meta.paper_id, target_chars=6000):
        if size + len(chunk.text) > budget and current:
            windows.append(current)
            current, size = [], 0
        current.append((chunk.section, chunk.text))
        size += len(chunk.text)
    if current:
        windows.append(current)

    digests: list[SectionDigest] = []
    for index, window in enumerate(windows, start=1):
        body = "\n\n".join(f"## {section}\n{text}" for section, text in window)
        prompt = (
            f"{_paper_header(meta)}\n\n"
            f"Below is part {index} of {len(windows)} of the paper's body.\n\n"
            f"{body}\n\n"
            "Summarize each section present in this excerpt. Return JSON: "
            '{"sections": [{"section": "...", "summary": "2-4 sentences", '
            '"key_points": ["..."], "terms": ["technical terms introduced"]}]}\n'
            "Preserve concrete numbers. Output raw JSON only.\n\n"
            f"{_language_block(language)}"
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
            log.warning("%s: section digest %d failed: %s", meta.paper_id, index, exc)
            continue

        for item in data.get("sections") or []:
            if not isinstance(item, dict):
                continue
            digests.append(
                SectionDigest(
                    section=as_text(item.get("section")) or "Untitled",
                    summary=as_text(item.get("summary")),
                    key_points=as_list(item.get("key_points"), limit=8),
                    terms=as_list(item.get("terms"), limit=12),
                )
            )
    return digests


def _build_summary(
    paper_id: str, data: dict, language: Language, model: str, digests: list[SectionDigest]
) -> PaperSummary:
    parsed_digests = digests or [
        SectionDigest(
            section=as_text(item.get("section")) or "Untitled",
            summary=as_text(item.get("summary")),
            key_points=as_list(item.get("key_points"), limit=8),
        )
        for item in (data.get("section_digests") or [])
        if isinstance(item, dict)
    ]
    return PaperSummary(
        paper_id=paper_id,
        language=language.value,
        model=model,
        one_liner=as_text(data.get("one_liner")),
        problem=as_text(data.get("problem")),
        motivation=as_text(data.get("motivation")),
        method=as_text(data.get("method")),
        contributions=as_list(data.get("contributions")),
        experiments=as_text(data.get("experiments")),
        results=as_list(data.get("results")),
        limitations=as_list(data.get("limitations")),
        concepts=as_list(data.get("concepts"), limit=30),
        topics=as_list(data.get("topics"), limit=6),
        datasets=as_list(data.get("datasets")),
        baselines=as_list(data.get("baselines")),
        open_questions=as_list(data.get("open_questions")),
        critique=as_text(data.get("critique")),
        section_digests=parsed_digests,
    )


def summarize_paper(
    store: PaperStore,
    paper_id: str,
    *,
    language: Language | None = None,
    client: LLMClient | None = None,
) -> PaperSummary:
    """Produce and persist the structured note for one paper.

    Short papers go through the model in a single pass. Anything larger than the
    context budget is digested section by section first, then reduced.
    """
    settings = store.settings
    language = language or settings.default_language
    client = client or LLMClient(settings.llm)

    meta = store.load_meta(paper_id)
    if meta is None:
        raise ValueError(f"unknown paper: {paper_id}")

    fulltext = store.load_fulltext(paper_id) or store.rebuild_fulltext(paper_id)
    if not fulltext and not meta.abstract:
        raise ValueError(f"{paper_id}: no text to summarize")

    # Leave room for the prompt scaffolding and the response itself.
    budget = int(settings.llm.max_context_chars * 0.75)
    digests: list[SectionDigest] = []

    if len(fulltext) <= budget:
        body = fulltext or "(full text unavailable; only the abstract is known)"
    else:
        log.info("%s: %d chars exceeds budget, digesting sections first", paper_id, len(fulltext))
        digests = _digest_sections(client, meta, store, language, budget)
        body = "\n\n".join(
            f"## {digest.section}\n{digest.summary}\n"
            + "\n".join(f"- {point}" for point in digest.key_points)
            for digest in digests
        )

    known = _known_topics(store)
    topic_hint = (
        f"\nTopic labels already used in this library, reuse them when they fit: "
        f"{', '.join(known)}.\n" if known else ""
    )

    prompt = (
        f"{_paper_header(meta)}\n\n"
        f"--- PAPER BODY ---\n{body}\n--- END PAPER BODY ---\n\n"
        f"{_SCHEMA}\n{topic_hint}\n{_language_block(language)}"
    )

    data = client.complete_json(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    summary = _build_summary(paper_id, data, language, settings.llm.model, digests)
    store.save_summary(summary, render_summary_md(meta, summary))
    return summary


def render_summary_md(meta: PaperMeta, summary: PaperSummary) -> str:
    """Render a note as the Markdown file a human actually reads."""

    def bullets(title: str, items: list[str]) -> list[str]:
        if not items:
            return []
        return [f"## {title}", "", *[f"- {item}" for item in items], ""]

    def block(title: str, text: str) -> list[str]:
        return [f"## {title}", "", text, ""] if text else []

    lines: list[str] = [f"# {meta.display_title}", ""]

    facts = [f"**{meta.author_line}**"]
    if meta.abs_url:
        facts.append(f"[arXiv:{meta.arxiv_id}]({meta.abs_url})")
    if meta.published:
        facts.append(meta.published[:10])
    if meta.categories:
        facts.append(", ".join(meta.categories[:4]))
    lines += [" · ".join(facts), ""]

    if summary.one_liner:
        lines += [f"> {summary.one_liner}", ""]
    if summary.topics:
        lines += ["`" + "` `".join(summary.topics) + "`", ""]

    lines += block("Problem", summary.problem)
    lines += block("Motivation", summary.motivation)
    lines += block("Method", summary.method)
    lines += bullets("Contributions", summary.contributions)
    lines += block("Experimental setup", summary.experiments)
    lines += bullets("Results", summary.results)
    lines += bullets("Limitations", summary.limitations)
    lines += block("Critique", summary.critique)
    lines += bullets("Open questions", summary.open_questions)

    reference = []
    if summary.datasets:
        reference.append(f"**Datasets** — {', '.join(summary.datasets)}")
    if summary.baselines:
        reference.append(f"**Baselines** — {', '.join(summary.baselines)}")
    if summary.concepts:
        reference.append(f"**Concepts** — {', '.join(summary.concepts)}")
    if reference:
        lines += ["## At a glance", "", *[f"- {item}" for item in reference], ""]

    if summary.section_digests:
        lines += ["## Section notes", ""]
        for digest in summary.section_digests:
            lines.append(f"### {digest.section}")
            lines.append("")
            if digest.summary:
                lines += [digest.summary, ""]
            for point in digest.key_points:
                lines.append(f"- {point}")
            lines.append("")

    lines += [
        "---",
        "",
        f"*Generated by surveyor using {summary.model} on {summary.generated_at[:10]}.*",
        "",
    ]
    return "\n".join(lines)
