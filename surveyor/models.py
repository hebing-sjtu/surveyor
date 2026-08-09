"""Data models persisted under ``papers/<paper_id>/``."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperMeta(BaseModel):
    """Bibliographic record for one paper."""

    paper_id: str
    arxiv_id: str | None = None
    version: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    published: str | None = None
    updated: str | None = None
    doi: str | None = None
    journal_ref: str | None = None
    comment: str | None = None
    abs_url: str | None = None
    pdf_url: str | None = None
    source_kind: Literal["arxiv", "local"] = "arxiv"
    # Surveys get a different kind of note: a taxonomy of a field rather than
    # one method's problem and results.
    kind: Literal["paper", "survey"] = "paper"
    has_source: bool = False
    # Free-form labels the user can add; topics are LLM-assigned.
    tags: list[str] = Field(default_factory=list)
    ingested_at: str = Field(default_factory=utcnow)

    @property
    def author_line(self) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{self.authors[0]} et al. ({len(self.authors)} authors)"

    @property
    def display_title(self) -> str:
        return self.title or self.paper_id


class SectionDigest(BaseModel):
    """Output of the map pass over one window of the paper."""

    section: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)


class PaperSummary(BaseModel):
    """Output of the reduce pass: the structured note for one paper."""

    paper_id: str
    language: str = "zh"
    model: str = ""
    generated_at: str = Field(default_factory=utcnow)

    one_liner: str = ""
    problem: str = ""
    motivation: str = ""
    method: str = ""
    contributions: list[str] = Field(default_factory=list)
    experiments: str = ""
    results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    # Concepts worth remembering, used to link papers together.
    concepts: list[str] = Field(default_factory=list)
    # Coarse research areas, used to group papers into topic digests.
    topics: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    baselines: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    # Anything the reader should be skeptical about.
    critique: str = ""
    section_digests: list[SectionDigest] = Field(default_factory=list)


class TaxonomyNode(BaseModel):
    """One branch of how a survey carves up its field."""

    name: str
    description: str = ""
    # Citation keys or arXiv ids the survey files under this branch.
    representative: list[str] = Field(default_factory=list)
    children: list["TaxonomyNode"] = Field(default_factory=list)

    def walk(self, depth: int = 1) -> list[tuple[int, "TaxonomyNode"]]:
        found = [(depth, self)]
        for child in self.children:
            found.extend(child.walk(depth + 1))
        return found


TaxonomyNode.model_rebuild()


class SurveyReference(BaseModel):
    """One entry from a survey's bibliography, with how the survey used it."""

    key: str
    title: str = ""
    year: str = ""
    arxiv_id: str | None = None
    # Citations in prose, which drive ranking, kept apart from citations that
    # only appear in comparison tables.
    citations: int = 0
    table_citations: int = 0
    sections: list[str] = Field(default_factory=list)
    in_library: bool = False

    @property
    def display(self) -> str:
        label = self.title or self.key
        return f"{label} ({self.year})" if self.year else label

    @property
    def primary_section(self) -> str:
        """The branch this work is most tellingly discussed under."""
        if not self.sections:
            return ""
        return " > ".join(self.sections[0].split(" > ")[-2:])


class SurveyNote(BaseModel):
    """The structured note for a survey: a map of a field, not of one method."""

    paper_id: str
    language: str = "zh"
    model: str = ""
    generated_at: str = Field(default_factory=utcnow)

    # Canonical field name, used to group several surveys of the same area.
    field_name: str = ""
    scope: str = ""
    taxonomy: list[TaxonomyNode] = Field(default_factory=list)
    key_concepts: list[str] = Field(default_factory=list)
    benchmarks: list[str] = Field(default_factory=list)
    # Landmark works in the order the survey presents them.
    milestones: list[str] = Field(default_factory=list)
    consensus: list[str] = Field(default_factory=list)
    open_challenges: list[str] = Field(default_factory=list)
    reading_path: list[str] = Field(default_factory=list)
    section_digests: list[SectionDigest] = Field(default_factory=list)

    def flat_taxonomy(self) -> list[tuple[int, TaxonomyNode]]:
        return [item for node in self.taxonomy for item in node.walk()]


class Chunk(BaseModel):
    """A retrievable slice of a paper."""

    paper_id: str
    chunk_id: str
    section: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.paper_id}#{self.section}" if self.section else self.paper_id


class PaperRecord(BaseModel):
    """Everything the agent knows about a paper, assembled from disk."""

    meta: PaperMeta
    summary: PaperSummary | None = None
    fulltext_chars: int = 0
