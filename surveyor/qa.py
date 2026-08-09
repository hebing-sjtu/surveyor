"""Question answering over the library, with citations back to the source."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import Language
from .llm import LLMClient
from .retrieval import Hit, build_index
from .store import PaperStore

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You answer questions about a researcher's personal library of machine-learning "
    "papers. You only use the excerpts provided. Every factual claim you make must "
    "carry a citation in the form [paper_id §section]. If the excerpts do not contain "
    "the answer, say plainly what is missing rather than filling the gap from memory. "
    "Distinguish clearly between what a paper claims and what it demonstrates."
)


@dataclass
class Answer:
    question: str
    text: str
    sources: list[str] = field(default_factory=list)
    used_papers: list[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        if not self.sources:
            return self.text
        return self.text + "\n\n**Sources**\n" + "\n".join(f"- {s}" for s in self.sources)


def _context_from_hits(hits: list[Hit], budget: int) -> tuple[str, list[str]]:
    blocks: list[str] = []
    sources: list[str] = []
    used = 0
    for hit in hits:
        chunk = hit.chunk
        header = f"[{chunk.paper_id} §{chunk.section}]"
        block = f"{header}\n{chunk.text}"
        if used + len(block) > budget:
            continue
        blocks.append(block)
        sources.append(header.strip("[]"))
        used += len(block)
    return "\n\n---\n\n".join(blocks), sources


def _titles_block(store: PaperStore, paper_ids: list[str]) -> str:
    lines = []
    for paper_id in paper_ids:
        meta = store.load_meta(paper_id)
        if meta:
            lines.append(f"- {paper_id}: {meta.display_title}")
    return "\n".join(lines)


def ask(
    store: PaperStore,
    question: str,
    *,
    paper_ids: list[str] | None = None,
    language: Language | None = None,
    top_k: int = 10,
    client: LLMClient | None = None,
) -> Answer:
    """Answer a question, optionally scoped to specific papers.

    Scoping to a single paper that fits in context skips retrieval entirely and
    reads the whole thing, which is both cheaper to reason about and more
    faithful than stitching together fragments.
    """
    settings = store.settings
    language = language or settings.default_language
    client = client or LLMClient(settings.llm)
    budget = int(settings.llm.max_context_chars * 0.7)

    if paper_ids and len(paper_ids) == 1:
        answer = _ask_single_paper(store, question, paper_ids[0], language, client, budget)
        if answer is not None:
            return answer

    index = build_index(store, paper_ids)
    hits = index.search(question, limit=top_k, paper_ids=set(paper_ids) if paper_ids else None)
    if not hits:
        return Answer(
            question=question,
            text=(
                "I could not find anything relevant in the library for that question. "
                "Try different wording, or add the paper first."
            ),
        )

    context, sources = _context_from_hits(hits, budget)
    used_papers = list(dict.fromkeys(hit.chunk.paper_id for hit in hits))

    prompt = (
        f"Papers these excerpts come from:\n{_titles_block(store, used_papers)}\n\n"
        f"--- EXCERPTS ---\n{context}\n--- END EXCERPTS ---\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above, citing as [paper_id §section]. "
        "When several papers disagree, say so and attribute each position.\n\n"
        f"{language.instruction}"
    )
    text = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return Answer(question=question, text=text, sources=sources, used_papers=used_papers)


def _ask_single_paper(
    store: PaperStore,
    question: str,
    paper_id: str,
    language: Language,
    client: LLMClient,
    budget: int,
) -> Answer | None:
    fulltext = store.load_fulltext(paper_id)
    if not fulltext or len(fulltext) > budget:
        return None

    meta = store.load_meta(paper_id)
    title = meta.display_title if meta else paper_id
    prompt = (
        f"Paper: {title} ({paper_id})\n\n"
        f"--- FULL TEXT ---\n{fulltext}\n--- END FULL TEXT ---\n\n"
        f"Question: {question}\n\n"
        "Answer from this paper alone, citing section names as [paper_id §section].\n\n"
        f"{language.instruction}"
    )
    text = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return Answer(question=question, text=text, sources=[paper_id], used_papers=[paper_id])


def compare(
    store: PaperStore,
    paper_ids: list[str],
    *,
    aspect: str = "",
    language: Language | None = None,
    client: LLMClient | None = None,
) -> str:
    """Compare several papers side by side, using their structured notes."""
    settings = store.settings
    language = language or settings.default_language
    client = client or LLMClient(settings.llm)

    blocks: list[str] = []
    for paper_id in paper_ids:
        meta = store.load_meta(paper_id)
        summary = store.load_summary(paper_id)
        if meta is None:
            continue
        if summary is None:
            blocks.append(f"### {paper_id}: {meta.display_title}\n{meta.abstract}")
            continue
        blocks.append(
            f"### {paper_id}: {meta.display_title}\n"
            f"One-liner: {summary.one_liner}\n"
            f"Problem: {summary.problem}\n"
            f"Method: {summary.method}\n"
            f"Results: {'; '.join(summary.results)}\n"
            f"Limitations: {'; '.join(summary.limitations)}\n"
            f"Datasets: {', '.join(summary.datasets)}\n"
            f"Baselines: {', '.join(summary.baselines)}"
        )

    if len(blocks) < 2:
        return "Need at least two papers with notes to compare."

    focus = f"Focus the comparison on: {aspect}\n" if aspect else ""
    prompt = (
        "\n\n".join(blocks) + "\n\n"
        f"{focus}"
        "Compare these papers. Cover: what problem each targets, how the methods "
        "genuinely differ (not just naming), what evidence each provides, where they "
        "agree and disagree, and which to prefer under which circumstances. "
        "Start with a compact comparison table, then the discussion. "
        "Be explicit when a comparison is not meaningful because the setups differ.\n\n"
        f"{language.instruction}"
    )
    return client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
