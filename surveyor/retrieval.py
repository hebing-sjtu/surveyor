"""BM25 retrieval over the library.

Pure Python and dependency-free on purpose: a personal reading list is a few
hundred papers at most, which BM25 handles in milliseconds without an embedding
service, an index server, or a network round trip before every question.
Chinese queries work because CJK runs are indexed as unigrams plus bigrams.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .models import Chunk
from .store import PaperStore

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*|\d+(?:\.\d+)?")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with", "we",
    "our", "their", "they", "can", "does", "do", "not", "но", "的", "了", "是",
}

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Latin words plus CJK unigrams and bigrams, lowercased, stopwords dropped."""
    lowered = text.lower()
    tokens = [token for token in _WORD.findall(lowered) if token not in _STOPWORDS]
    for run in _CJK_RUN.findall(lowered):
        tokens.extend(char for char in run if char not in _STOPWORDS)
        tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.term_freqs: list[Counter[str]] = []
        self.lengths: list[int] = []
        document_freq: Counter[str] = Counter()

        for chunk in chunks:
            tokens = tokenize(f"{chunk.section} {chunk.text}")
            counts = Counter(tokens)
            self.term_freqs.append(counts)
            self.lengths.append(len(tokens))
            document_freq.update(counts.keys())

        total = len(chunks)
        self.avg_length = (sum(self.lengths) / total) if total else 0.0
        self.idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_freq.items()
        }

    def search(
        self, query: str, limit: int = 8, paper_ids: set[str] | None = None
    ) -> list[Hit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        scored: list[Hit] = []
        for index, chunk in enumerate(self.chunks):
            if paper_ids is not None and chunk.paper_id not in paper_ids:
                continue
            counts = self.term_freqs[index]
            length = self.lengths[index] or 1
            score = 0.0
            for token in query_tokens:
                freq = counts.get(token)
                if not freq:
                    continue
                idf = self.idf.get(token, 0.0)
                denominator = freq + K1 * (1 - B + B * length / (self.avg_length or 1))
                score += idf * freq * (K1 + 1) / denominator
            if score > 0:
                scored.append(Hit(chunk=chunk, score=score))

        scored.sort(key=lambda hit: hit.score, reverse=True)
        return scored[:limit]


def build_index(store: PaperStore, paper_ids: list[str] | None = None) -> BM25Index:
    """Index paper bodies plus their summaries.

    Summaries are indexed too so that library-level questions ("which papers use
    diffusion for video?") match even when the phrasing never appears verbatim in
    any paper body.
    """
    chunks: list[Chunk] = []
    for paper_id in paper_ids if paper_ids is not None else store.list_ids():
        summary = store.load_summary(paper_id)
        if summary:
            digest = "\n".join(
                part
                for part in (
                    summary.one_liner,
                    summary.problem,
                    summary.method,
                    " ".join(summary.contributions),
                    " ".join(summary.results),
                    " ".join(summary.concepts),
                    " ".join(summary.topics),
                )
                if part
            )
            if digest:
                chunks.append(
                    Chunk(
                        paper_id=paper_id,
                        chunk_id=f"{paper_id}::summary",
                        section="Summary",
                        text=digest,
                    )
                )
        chunks.extend(store.chunks(paper_id))
    return BM25Index(chunks)


def group_by_paper(hits: list[Hit]) -> dict[str, list[Hit]]:
    grouped: dict[str, list[Hit]] = {}
    for hit in hits:
        grouped.setdefault(hit.chunk.paper_id, []).append(hit)
    return grouped
