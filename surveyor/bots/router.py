"""Platform-independent command handling.

Feishu and WeCom differ only in transport and encryption; the conversation
itself is identical, so it lives here and is also reachable from the CLI via
``paper chat``.

Commands are parsed leniently: a message that is just an arXiv link is treated
as "add this paper", and a message with no recognised command is treated as a
question about the library.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Language
from ..ingest.resolve import extract_refs
from ..knowledge import slugify, topic_index, write_overview, write_topic_digest
from ..llm import LLMClient
from ..pipeline import harvest_survey, ingest_inputs
from ..qa import ask as ask_library
from ..qa import compare as compare_papers
from ..store import PaperStore
from ..survey import (
    core_references,
    group_surveys_by_field,
    merge_surveys,
    reading_list,
)

log = logging.getLogger(__name__)


@dataclass
class BotContext:
    platform: str
    user_id: str
    chat_id: str = ""
    is_group: bool = False

    @property
    def pref_key(self) -> str:
        return f"{self.platform}:{self.user_id}"


@dataclass
class BotReply:
    """``ack`` goes out immediately; ``work`` runs afterwards and is sent later.

    Both platforms expect a callback response within seconds, but ingesting and
    summarizing a paper takes minutes, so slow commands always defer.
    """

    ack: str | None = None
    work: Callable[[], str] | None = None


HELP_TEXT = """**Surveyor**

*Add papers*
`add 2503.07598` — arXiv id, URL, or several at once
Just sending an arXiv link also works.

*Read*
`list` — everything in the library
`find <words>` — search titles and notes
`summary <id or title>` — the structured note
`ask <question>` — ask across the whole library
`paper <id> <question>` — ask about one paper
`compare <id> <id> | <aspect>` — side by side

*Surveys*
`survey add <id>` — import a survey: taxonomy + reference list
`surveys` — surveys in the library
`survey <id>` — one survey's taxonomy
`refs <id>` — what it cites, most-cited first
`harvest <id>` — ingest the papers it cites most
`field [name]` — reconcile several surveys of one field
`core` — references several surveys agree on

*Synthesize*
`topics` — topic list
`topic <name>` — the digest for one area
`overview` — what the library adds up to

*Settings*
`lang zh` / `lang en` / `lang bilingual`
`status` — library and model health
"""

_COMMAND_ALIASES = {
    "add": "add", "ingest": "add", "添加": "add", "导入": "add",
    "list": "list", "ls": "list", "列表": "list", "papers": "list",
    "find": "find", "search": "find", "搜索": "find", "查找": "find",
    "summary": "summary", "sum": "summary", "总结": "summary", "摘要": "summary",
    "ask": "ask", "q": "ask", "问": "ask", "提问": "ask",
    "paper": "paper", "论文": "paper",
    "compare": "compare", "对比": "compare", "比较": "compare",
    "topics": "topics", "主题": "topics",
    "topic": "topic", "专题": "topic",
    "survey": "survey", "综述": "survey",
    "surveys": "surveys", "综述列表": "surveys",
    "refs": "refs", "references": "refs", "参考文献": "refs",
    "harvest": "harvest", "采集": "harvest", "导入引用": "harvest",
    "field": "field", "领域": "field",
    "core": "core", "核心": "core", "核心文献": "core",
    # 综述 means "survey paper", so it belongs to `survey`, not the library overview.
    "overview": "overview", "概览": "overview", "总览": "overview",
    "lang": "lang", "language": "lang", "语言": "lang",
    "status": "status", "状态": "status",
    "help": "help", "帮助": "help", "?": "help", "？": "help",
}


class Preferences:
    """Per-user settings, persisted so a bot restart does not forget them."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, dict] = {}
        if path.is_file():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._data = {}

    def language(self, key: str, default: Language) -> Language:
        return Language.parse((self._data.get(key) or {}).get("language"), default)

    def set_language(self, key: str, language: Language) -> None:
        self._data.setdefault(key, {})["language"] = language.value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class Router:
    def __init__(self, store: PaperStore | None = None) -> None:
        self.store = store or PaperStore()
        self.prefs = Preferences(self.store.settings.state_dir / "prefs.json")

    # ------------------------------------------------------------------ util
    def _language(self, context: BotContext) -> Language:
        return self.prefs.language(context.pref_key, self.store.settings.default_language)

    def _client(self) -> LLMClient:
        return LLMClient(self.store.settings.llm)

    def _resolve_or_explain(self, query: str) -> tuple[str | None, str]:
        matches = self.store.resolve(query)
        if not matches:
            return None, f"No paper matches `{query}`. Try `list` to see what is loaded."
        if len(matches) > 1:
            listing = "\n".join(f"- `{pid}`" for pid in matches[:10])
            return None, f"`{query}` matches several papers:\n{listing}"
        return matches[0], ""

    # --------------------------------------------------------------- dispatch
    def handle(self, message: str, context: BotContext) -> BotReply:
        text = _clean_message(message)
        if not text:
            return BotReply(ack=HELP_TEXT)

        head, _, rest = text.partition(" ")
        command = _COMMAND_ALIASES.get(head.strip().lower().lstrip("/"))
        argument = rest.strip()

        if command is None:
            # A bare arXiv reference means "add this"; anything else is a question.
            if extract_refs(text):
                command, argument = "add", text
            else:
                command, argument = "ask", text

        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            return BotReply(ack=HELP_TEXT)
        try:
            return handler(argument, context)
        except Exception as exc:
            log.exception("command %s failed", command)
            return BotReply(ack=f"Something went wrong: {exc}")

    # --------------------------------------------------------------- commands
    def _cmd_help(self, _argument: str, _context: BotContext) -> BotReply:
        return BotReply(ack=HELP_TEXT)

    def _cmd_add(self, argument: str, context: BotContext) -> BotReply:
        return self._add(argument, context, kind=None)

    def _add(self, argument: str, context: BotContext, kind: str | None) -> BotReply:
        refs = extract_refs(argument)
        if not refs:
            return BotReply(ack="I could not find an arXiv id or link in that message.")
        language = self._language(context)
        names = ", ".join(ref.canonical for ref in refs)
        noun = "survey" if kind == "survey" else "paper"

        def work() -> str:
            results = ingest_inputs(
                self.store, [argument], summarize=True, language=language, kind=kind
            )
            if not results:
                return "Nothing was ingested."
            lines = ["**Ingestion finished**", ""]
            for result in results:
                meta = self.store.load_meta(result.paper_id)
                title = (meta.display_title if meta else result.paper_id)[:100]
                lines.append(f"- `{result.paper_id}` {result.status()} — {title}")
                lines.extend(f"    - {error}" for error in result.errors)

            surveys = [result for result in results if result.is_survey]
            done = [result for result in results if result.summarized]
            if surveys:
                lines += [
                    "",
                    (
                        f"Read the taxonomy with `survey {surveys[0].paper_id}`, or "
                        f"pull in what it cites with `harvest {surveys[0].paper_id}`."
                    ),
                ]
            elif done:
                lines += ["", f"Ask about them with `paper {done[0].paper_id} <question>`."]
            return "\n".join(lines)

        return BotReply(
            ack=f"Fetching {len(refs)} {noun}(s): {names}\nThis takes a couple of minutes.",
            work=work,
        )

    def _cmd_list(self, argument: str, _context: BotContext) -> BotReply:
        records = list(self.store.iter_records())
        if not records:
            return BotReply(ack="The library is empty. Add a paper with `add <arxiv id>`.")
        records.sort(key=lambda record: record.meta.published or "", reverse=True)
        limit = 40 if not argument.isdigit() else int(argument)
        lines = [f"**{len(records)} papers**", ""]
        for record in records[:limit]:
            mark = "" if record.summary else " (no note yet)"
            date = (record.meta.published or "")[:7]
            lines.append(f"- `{record.meta.paper_id}` {date} {record.meta.display_title}{mark}")
        if len(records) > limit:
            lines.append(f"\n…and {len(records) - limit} more.")
        return BotReply(ack="\n".join(lines))

    def _cmd_find(self, argument: str, _context: BotContext) -> BotReply:
        if not argument:
            return BotReply(ack="Usage: `find <words>`")
        needle = argument.lower()
        hits = []
        for record in self.store.iter_records():
            haystack = " ".join(
                [
                    record.meta.title,
                    record.meta.abstract,
                    record.summary.one_liner if record.summary else "",
                    " ".join(record.summary.concepts) if record.summary else "",
                    " ".join(record.summary.topics) if record.summary else "",
                ]
            ).lower()
            if needle in haystack:
                hits.append(record)
        if not hits:
            return BotReply(ack=f"Nothing matches `{argument}`. Try `ask {argument}` instead.")
        lines = [f"**{len(hits)} match(es)**", ""]
        lines.extend(
            f"- `{record.meta.paper_id}` {record.meta.display_title}"
            for record in hits[:20]
        )
        return BotReply(ack="\n".join(lines))

    def _cmd_summary(self, argument: str, _context: BotContext) -> BotReply:
        if not argument:
            return BotReply(ack="Usage: `summary <id or title>`")
        paper_id, problem = self._resolve_or_explain(argument)
        if paper_id is None:
            return BotReply(ack=problem)
        path = self.store.paper_dir(paper_id) / "summary.md"
        if not path.is_file():
            return BotReply(ack=f"`{paper_id}` has no note yet. Run `add {paper_id}` to build one.")
        return BotReply(ack=path.read_text(encoding="utf-8"))

    def _cmd_ask(self, argument: str, context: BotContext) -> BotReply:
        if not argument:
            return BotReply(ack="Usage: `ask <question>`")
        language = self._language(context)

        def work() -> str:
            answer = ask_library(self.store, argument, language=language)
            return answer.as_markdown()

        return BotReply(ack="Thinking…", work=work)

    def _cmd_paper(self, argument: str, context: BotContext) -> BotReply:
        parts = argument.split(None, 1)
        if len(parts) < 2:
            return BotReply(ack="Usage: `paper <id> <question>`")
        target, question = parts
        paper_id, problem = self._resolve_or_explain(target)
        if paper_id is None:
            return BotReply(ack=problem)
        language = self._language(context)

        def work() -> str:
            answer = ask_library(
                self.store, question, paper_ids=[paper_id], language=language
            )
            return answer.as_markdown()

        return BotReply(ack=f"Reading `{paper_id}`…", work=work)

    def _cmd_compare(self, argument: str, context: BotContext) -> BotReply:
        body, _, aspect = argument.partition("|")
        try:
            tokens = shlex.split(body)
        except ValueError:
            tokens = body.split()
        if len(tokens) < 2:
            return BotReply(ack="Usage: `compare <id> <id> [| aspect]`")

        resolved: list[str] = []
        for token in tokens:
            paper_id, problem = self._resolve_or_explain(token)
            if paper_id is None:
                return BotReply(ack=problem)
            resolved.append(paper_id)
        language = self._language(context)

        def work() -> str:
            return compare_papers(
                self.store, resolved, aspect=aspect.strip(), language=language
            )

        return BotReply(ack=f"Comparing {', '.join(f'`{p}`' for p in resolved)}…", work=work)

    def _cmd_topics(self, _argument: str, _context: BotContext) -> BotReply:
        grouped = topic_index(self.store)
        if not grouped:
            return BotReply(ack="No topics yet: summarize some papers first.")
        lines = ["**Topics**", ""]
        for topic, records in grouped.items():
            lines.append(f"- **{topic}** ({len(records)}) — `topic {topic}`")
        return BotReply(ack="\n".join(lines))

    def _cmd_topic(self, argument: str, context: BotContext) -> BotReply:
        if not argument:
            return BotReply(ack="Usage: `topic <name>` (see `topics`)")
        path = self.store.settings.knowledge_dir / "topics" / f"{slugify(argument)}.md"
        if path.is_file():
            return BotReply(ack=path.read_text(encoding="utf-8"))

        grouped = topic_index(self.store)
        match = next(
            (topic for topic in grouped if slugify(topic) == slugify(argument)), None
        )
        if match is None:
            available = ", ".join(grouped) or "none yet"
            return BotReply(ack=f"No topic called `{argument}`. Available: {available}")
        language = self._language(context)
        records = grouped[match]

        def work() -> str:
            written = write_topic_digest(
                self.store, match, records, language=language
            )
            return written.read_text(encoding="utf-8")

        return BotReply(ack=f"Writing the digest for **{match}** ({len(records)} papers)…", work=work)

    # ---------------------------------------------------------------- surveys
    def _cmd_surveys(self, _argument: str, _context: BotContext) -> BotReply:
        survey_ids = self.store.list_surveys()
        if not survey_ids:
            return BotReply(ack="No surveys yet. Import one with `survey add <arxiv id>`.")
        lines = [f"**{len(survey_ids)} survey(s)**", ""]
        for paper_id in survey_ids:
            note = self.store.load_survey(paper_id)
            references = self.store.load_references(paper_id)
            with_id = sum(1 for ref in references if ref.arxiv_id)
            field_name = note.field_name if note else "no note yet"
            lines.append(
                f"- `{paper_id}` **{field_name}** — "
                f"{len(note.flat_taxonomy()) if note else 0} branches, "
                f"{with_id}/{len(references)} refs on arXiv"
            )
        lines += ["", "Use `survey <id>`, `refs <id>`, `harvest <id>` or `field`."]
        return BotReply(ack="\n".join(lines))

    def _cmd_survey(self, argument: str, context: BotContext) -> BotReply:
        # `survey add <ids>` doubles as the import command.
        head, _, rest = argument.partition(" ")
        if head.strip().lower() in ("add", "import", "添加", "导入"):
            return self._add(rest.strip(), context, kind="survey")

        if not argument:
            return self._cmd_surveys("", context)
        paper_id, problem = self._resolve_or_explain(argument)
        if paper_id is None:
            return BotReply(ack=problem)
        path = self.store.paper_dir(paper_id) / "survey.md"
        if not path.is_file():
            return BotReply(
                ack=f"`{paper_id}` has no survey note. Import it with "
                f"`survey add {paper_id}`."
            )
        return BotReply(ack=path.read_text(encoding="utf-8"))

    def _cmd_refs(self, argument: str, _context: BotContext) -> BotReply:
        parts = argument.split()
        if not parts:
            return BotReply(ack="Usage: `refs <survey id> [count]`")
        limit = int(parts[-1]) if parts[-1].isdigit() and len(parts) > 1 else 15
        target = " ".join(parts[:-1]) if parts[-1].isdigit() and len(parts) > 1 else argument

        paper_id, problem = self._resolve_or_explain(target)
        if paper_id is None:
            return BotReply(ack=problem)
        references = self.store.load_references(paper_id)
        if not references:
            return BotReply(ack=f"No bibliography recorded for `{paper_id}`.")

        picks = reading_list(references, limit=limit, only_arxiv=True)
        lines = [
            f"**Most-cited references in `{paper_id}`**",
            "",
            *[
                f"- {'✓' if ref.in_library else '·'} `{ref.arxiv_id}` ×{ref.citations} — "
                f"{(ref.title or ref.key)[:70]}"
                for ref in picks
            ],
            "",
            (
                f"{sum(1 for r in references if r.arxiv_id)}/{len(references)} "
                "references have arXiv sources. `harvest` ingests the top ones."
            ),
        ]
        return BotReply(ack="\n".join(lines))

    def _cmd_harvest(self, argument: str, context: BotContext) -> BotReply:
        parts = argument.split()
        if not parts:
            return BotReply(ack="Usage: `harvest <survey id> [count]`")
        limit = 10
        if parts[-1].isdigit() and len(parts) > 1:
            limit = min(int(parts[-1]), 50)
            parts = parts[:-1]
        paper_id, problem = self._resolve_or_explain(" ".join(parts))
        if paper_id is None:
            return BotReply(ack=problem)
        language = self._language(context)

        def work() -> str:
            results = harvest_survey(
                self.store, paper_id, limit=limit, language=language
            )
            if not results:
                return "Every arXiv reference from that survey is already in the library."
            lines = [f"**Harvested from `{paper_id}`**", ""]
            for result in results:
                meta = self.store.load_meta(result.paper_id)
                title = (meta.display_title if meta else result.paper_id)[:80]
                lines.append(f"- `{result.paper_id}` {result.status()} — {title}")
            return "\n".join(lines)

        return BotReply(
            ack=f"Ingesting up to {limit} papers cited by `{paper_id}`. "
            "This takes a while.",
            work=work,
        )

    def _cmd_field(self, argument: str, context: BotContext) -> BotReply:
        grouped = group_surveys_by_field(self.store)
        if not grouped:
            return BotReply(ack="No surveys with notes yet. Try `survey add <arxiv id>`.")

        if not argument:
            if len(grouped) > 1:
                lines = ["**Fields covered by surveys**", ""]
                for name, ids in grouped.items():
                    lines.append(f"- **{name}** ({len(ids)} survey(s)) — `field {name}`")
                return BotReply(ack="\n".join(lines))
            argument = next(iter(grouped))

        match = next(
            (name for name in grouped if slugify(name) == slugify(argument)), None
        )
        if match is None:
            return BotReply(
                ack=f"No field called `{argument}`. Known: {', '.join(grouped)}"
            )

        path = self.store.settings.knowledge_dir / "fields" / f"{slugify(match)}.md"
        if path.is_file():
            return BotReply(ack=path.read_text(encoding="utf-8"))

        survey_ids = grouped[match]
        language = self._language(context)

        def work() -> str:
            return merge_surveys(
                self.store, survey_ids, field_name=match, language=language
            ).read_text(encoding="utf-8")

        return BotReply(
            ack=f"Reconciling {len(survey_ids)} survey(s) for **{match}**…", work=work
        )

    def _cmd_core(self, argument: str, _context: BotContext) -> BotReply:
        survey_ids = self.store.list_surveys()
        if len(survey_ids) < 2:
            return BotReply(
                ack="Need at least two surveys to find shared references. "
                f"The library has {len(survey_ids)}."
            )
        shared = core_references(self.store, survey_ids, min_surveys=2)
        if not shared:
            return BotReply(ack="No references are cited by more than one survey.")
        limit = int(argument) if argument.strip().isdigit() else 20
        lines = [
            f"**Core references** — cited by several of the {len(survey_ids)} surveys",
            "",
            *[
                f"- {'✓' if ref.in_library else '·'} {count} surveys, ×{ref.citations} "
                f"cites — `{ref.arxiv_id}` {(ref.title or ref.key)[:60]}"
                for ref, count, _ids in shared[:limit]
            ],
        ]
        return BotReply(ack="\n".join(lines))

    def _cmd_overview(self, _argument: str, context: BotContext) -> BotReply:
        language = self._language(context)

        def work() -> str:
            return write_overview(self.store, language=language).read_text(encoding="utf-8")

        return BotReply(ack="Synthesizing the library…", work=work)

    def _cmd_lang(self, argument: str, context: BotContext) -> BotReply:
        if not argument:
            current = self._language(context)
            return BotReply(ack=f"Current language: `{current.value}`. Use `lang zh|en|bilingual`.")
        default = self.store.settings.default_language
        language = Language.parse(argument, default)
        if language.value != argument.strip().lower() and Language.parse(argument, default) == default:
            valid = ", ".join(item.value for item in Language)
            return BotReply(ack=f"Unknown language `{argument}`. Choose one of: {valid}")
        self.prefs.set_language(context.pref_key, language)
        return BotReply(ack=f"Replies to you will now be in `{language.value}`.")

    def _cmd_status(self, _argument: str, context: BotContext) -> BotReply:
        records = list(self.store.iter_records())
        summarized = sum(1 for record in records if record.summary)
        client = self._client()
        settings = self.store.settings
        lines = [
            "**Status**",
            "",
            f"- Papers: {len(records)} ({summarized} with notes)",
            f"- Topics: {len(topic_index(self.store))}",
            f"- Model: `{settings.llm.model}` at `{settings.llm.base_url}`",
            f"- API key: {'configured' if client.is_configured else 'MISSING'}",
            f"- Your language: `{self._language(context).value}`",
        ]
        return BotReply(ack="\n".join(lines))


_MENTION = re.compile(r"@_user_\d+|@[\w\u4e00-\u9fff.-]+")


def _clean_message(message: str) -> str:
    """Strip @-mentions the platforms leave in group messages."""
    return _MENTION.sub(" ", message or "").strip()


def split_message(text: str, limit: int) -> list[str]:
    """Break a reply into platform-sized pieces at paragraph or line boundaries."""
    if len(text.encode("utf-8")) <= limit:
        return [text]

    pieces: list[str] = []
    current = ""
    for block in text.split("\n"):
        candidate = f"{current}\n{block}" if current else block
        if len(candidate.encode("utf-8")) > limit and current:
            pieces.append(current)
            current = block
        else:
            current = candidate
        # A single line longer than the limit still has to be cut somewhere.
        while len(current.encode("utf-8")) > limit:
            cut = limit
            while cut > 0 and len(current[:cut].encode("utf-8")) > limit:
                cut -= 32
            pieces.append(current[:cut])
            current = current[cut:]
    if current.strip():
        pieces.append(current)
    return pieces
