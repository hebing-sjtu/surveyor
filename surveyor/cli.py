"""Command line interface: ``surveyor <command>``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.table import Table

from .config import Language, get_settings
from .knowledge import (
    slugify,
    topic_index,
    write_all_topic_digests,
    write_glossary,
    write_overview,
)
from .llm import LLMClient
from .pipeline import (
    adopt_directory,
    harvest_survey,
    ingest_inputs,
    rebuild_all,
    source_stats,
)
from .qa import ask as ask_library
from .qa import compare as compare_papers
from .store import PaperStore, write_index
from .summarize import summarize_paper
from .survey import (
    collect_references,
    core_references,
    group_surveys_by_field,
    merge_surveys,
    reading_list,
    resolve_missing_ids,
    summarize_survey,
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "Read, summarize and interrogate a personal arXiv reading list. "
        "Run `surveyor gui` for the browser app."
    ),
)
console = Console()


def _setup_logging(verbose: bool, default: int = logging.WARNING) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else default,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _store() -> PaperStore:
    return PaperStore(get_settings())


def _language(value: str | None) -> Language | None:
    if not value:
        return None
    return Language.parse(value, get_settings().default_language)


def _progress(message: str) -> None:
    console.print(f"  [dim]{message}[/dim]")


def _resolve_or_exit(store: PaperStore, query: str) -> str:
    matches = store.resolve(query)
    if not matches:
        console.print(f"[red]No paper matches[/red] {query!r}. Try `paper list`.")
        raise typer.Exit(1)
    if len(matches) > 1:
        console.print(f"[yellow]{query!r} is ambiguous:[/yellow]")
        for paper_id in matches:
            console.print(f"  {paper_id}")
        raise typer.Exit(1)
    return matches[0]


def _report(results: list) -> None:
    if not results:
        console.print("[yellow]Nothing to ingest: no arXiv id or URL found.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Paper")
    table.add_column("Status")
    table.add_column("Title", overflow="fold")
    for result in results:
        colour = {
            "failed": "red",
            "skipped": "dim",
            "summarized": "green",
            "survey": "cyan",
        }.get(result.status(), "yellow")
        table.add_row(result.paper_id, f"[{colour}]{result.status()}[/{colour}]", result.title)
    console.print(table)
    for result in results:
        for error in result.errors:
            console.print(f"[red]{result.paper_id}:[/red] {error}")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.command()
def add(
    inputs: list[str] = typer.Argument(..., help="arXiv ids, URLs, or a path to a list file"),
    no_summary: bool = typer.Option(False, "--no-summary", help="Download and convert only"),
    force: bool = typer.Option(False, "--force", help="Re-download and re-summarize"),
    lang: Optional[str] = typer.Option(None, "--lang", help="zh, en or bilingual"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Add papers from ids, URLs, or a list file."""
    _setup_logging(verbose)
    store = _store()
    if not no_summary and not LLMClient(store.settings.llm).is_configured:
        console.print(
            f"[yellow]No API key in {store.settings.llm.api_key_env}; "
            "downloading without summarizing.[/yellow]"
        )
        no_summary = True

    results = ingest_inputs(
        store,
        inputs,
        summarize=not no_summary,
        force=force,
        language=_language(lang),
        progress=_progress,
    )
    _report(results)


@app.command()
def adopt(
    directories: list[Path] = typer.Argument(..., help="Existing extracted source folders"),
    copy: bool = typer.Option(False, "--copy", help="Copy instead of moving"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Bring already-extracted LaTeX folders into the library."""
    _setup_logging(verbose)
    store = _store()
    summarize = not no_summary and LLMClient(store.settings.llm).is_configured
    results = []
    for directory in directories:
        if not directory.is_dir():
            console.print(f"[red]Not a directory:[/red] {directory}")
            continue
        results.append(
            adopt_directory(
                store,
                directory,
                move=not copy,
                summarize=summarize,
                language=_language(lang),
                progress=_progress,
            )
        )
    write_index(store)
    _report(results)


@app.command()
def migrate(
    pattern: str = typer.Option("arXiv-*", "--pattern", help="Glob for folders to adopt"),
    copy: bool = typer.Option(False, "--copy"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Adopt every loose source folder sitting in the library root."""
    _setup_logging(verbose)
    store = _store()
    root = store.settings.root
    candidates = [
        path
        for path in sorted(root.glob(pattern))
        if path.is_dir() and path != store.root and store.root not in path.parents
    ]
    if not candidates:
        console.print(f"No folders match {pattern!r} in {root}.")
        return

    console.print(f"Adopting {len(candidates)} folder(s).")
    summarize = not no_summary and LLMClient(store.settings.llm).is_configured
    results = [
        adopt_directory(
            store,
            path,
            move=not copy,
            summarize=summarize,
            language=_language(lang),
            progress=_progress,
        )
        for path in candidates
    ]
    write_index(store)
    _report(results)


# ---------------------------------------------------------------------------
# Surveys
# ---------------------------------------------------------------------------

survey_app = typer.Typer(
    no_args_is_help=True,
    help="Import surveys and use them to map out a field.",
)
app.add_typer(survey_app, name="survey")


@survey_app.command("add")
def survey_add(
    inputs: list[str] = typer.Argument(..., help="arXiv ids, URLs, or a list file"),
    force: bool = typer.Option(False, "--force", help="Re-download and re-analyze"),
    no_summary: bool = typer.Option(
        False, "--no-summary", help="References and structure only, no model calls"
    ),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Import surveys: extract their taxonomy and reference list."""
    _setup_logging(verbose)
    store = _store()
    summarize = not no_summary
    if summarize and not LLMClient(store.settings.llm).is_configured:
        console.print(
            f"[yellow]No API key in {store.settings.llm.api_key_env}; extracting "
            "references and structure without the taxonomy note.[/yellow]"
        )
        summarize = False

    results = ingest_inputs(
        store,
        inputs,
        summarize=summarize,
        force=force,
        language=_language(lang),
        kind="survey",
        progress=_progress,
    )
    _report(results)
    if results and not summarize:
        console.print(
            "\nReferences are indexed. `paper survey refs <id>` and `paper survey core` "
            "work now; run `paper survey reanalyze` once a key is set for the taxonomy."
        )


@survey_app.command("adopt")
def survey_adopt(
    directories: list[Path] = typer.Argument(..., help="Extracted survey source folders"),
    copy: bool = typer.Option(False, "--copy", help="Copy instead of moving"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Import surveys from LaTeX folders already on disk."""
    _setup_logging(verbose)
    store = _store()
    summarize = not no_summary and LLMClient(store.settings.llm).is_configured
    results = []
    for directory in directories:
        if not directory.is_dir():
            console.print(f"[red]Not a directory:[/red] {directory}")
            continue
        results.append(
            adopt_directory(
                store,
                directory,
                move=not copy,
                summarize=summarize,
                language=_language(lang),
                kind="survey",
                progress=_progress,
            )
        )
    write_index(store)
    _report(results)


@survey_app.command("list")
def survey_list() -> None:
    """List the surveys in the library."""
    store = _store()
    survey_ids = store.list_surveys()
    if not survey_ids:
        console.print("No surveys yet. Try `paper survey add <arxiv id>`.")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Paper")
    table.add_column("Field", overflow="fold")
    table.add_column("Branches", justify="right")
    table.add_column("Refs", justify="right")
    table.add_column("arXiv refs", justify="right")
    table.add_column("In library", justify="right")
    for paper_id in survey_ids:
        note = store.load_survey(paper_id)
        references = store.load_references(paper_id)
        with_id = [ref for ref in references if ref.arxiv_id]
        table.add_row(
            paper_id,
            note.field_name if note else "[yellow]no note[/yellow]",
            str(len(note.flat_taxonomy())) if note else "-",
            str(len(references)),
            str(len(with_id)),
            str(sum(1 for ref in with_id if ref.in_library)),
        )
    console.print(table)


@survey_app.command("show")
def survey_show(
    query: str = typer.Argument(...),
    raw: bool = typer.Option(False, "--raw"),
) -> None:
    """Print a survey's note and taxonomy."""
    store = _store()
    paper_id = _resolve_or_exit(store, query)
    path = store.paper_dir(paper_id) / "survey.md"
    if not path.is_file():
        console.print(
            f"[yellow]{paper_id} has no survey note. "
            f"Run `paper survey reanalyze {paper_id}`.[/yellow]"
        )
        raise typer.Exit(1)
    text = path.read_text(encoding="utf-8")
    console.print(text if raw else Markdown(text))


@survey_app.command("reanalyze")
def survey_reanalyze(
    query: Optional[str] = typer.Argument(None, help="Survey id, or all surveys"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Rebuild survey notes, e.g. after improving the converter."""
    _setup_logging(verbose)
    store = _store()
    targets = [_resolve_or_exit(store, query)] if query else store.list_surveys()
    if not targets:
        console.print("No surveys to analyze.")
        return
    for paper_id in targets:
        try:
            with console.status(f"Analyzing {paper_id}…"):
                note = summarize_survey(store, paper_id, language=_language(lang))
            console.print(
                f"[green]{paper_id}[/green] — {note.field_name}, "
                f"{len(note.flat_taxonomy())} taxonomy branches"
            )
        except Exception as exc:
            console.print(f"[red]{paper_id} failed:[/red] {exc}")
    write_index(store)


@survey_app.command("refs")
def survey_refs(
    query: str = typer.Argument(...),
    limit: int = typer.Option(30, "--limit", "-n"),
    section: Optional[str] = typer.Option(None, "--section", help="Only one taxonomy branch"),
    missing: bool = typer.Option(False, "--missing", help="Only ones not in the library"),
    resolve: bool = typer.Option(
        False,
        "--resolve",
        help="Search arXiv by title for references that cite only a venue version",
    ),
) -> None:
    """List a survey's references, ranked by how often it cites them."""
    store = _store()
    paper_id = _resolve_or_exit(store, query)
    references = store.load_references(paper_id) or collect_references(store, paper_id)
    if not references:
        console.print(f"No bibliography found for {paper_id}.")
        return

    if resolve:
        with console.status("Searching arXiv by title…"):
            found = resolve_missing_ids(store, paper_id, limit=max(limit, 30))
        console.print(f"Resolved {found} additional arXiv id(s).")
        references = store.load_references(paper_id)
    picks = reading_list(
        references, limit=limit, section=section, only_arxiv=True, only_missing=missing
    )
    if not picks:
        console.print("Nothing matched. Try without --missing or --section.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Prose", justify="right")
    table.add_column("Table", justify="right")
    table.add_column("arXiv")
    table.add_column("In lib")
    table.add_column("Reference", max_width=48, overflow="fold")
    table.add_column("Discussed under", max_width=34, overflow="fold")
    for reference in picks:
        table.add_row(
            str(reference.citations),
            str(reference.table_citations) or "",
            reference.arxiv_id or "-",
            "yes" if reference.in_library else "",
            (reference.title or reference.key)[:110],
            reference.primary_section,
        )
    console.print(table)
    console.print(
        f"\n{len(references)} cited references, "
        f"{sum(1 for r in references if r.arxiv_id)} with arXiv sources. "
        f"[dim]Prose = cited in the text, Table = only in comparison tables.[/dim]"
    )


@survey_app.command("harvest")
def survey_harvest(
    query: str = typer.Argument(...),
    limit: int = typer.Option(10, "--limit", "-n", help="How many papers to ingest"),
    section: Optional[str] = typer.Option(None, "--section"),
    resolve: bool = typer.Option(False, "--resolve", help="Search arXiv by title first"),
    no_summary: bool = typer.Option(False, "--no-summary"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ingest the papers a survey cites most."""
    _setup_logging(verbose)
    store = _store()
    paper_id = _resolve_or_exit(store, query)
    if resolve:
        with console.status("Searching arXiv by title…"):
            found = resolve_missing_ids(store, paper_id, limit=max(limit * 3, 30))
        console.print(f"Resolved {found} additional arXiv id(s).")
    summarize = not no_summary and LLMClient(store.settings.llm).is_configured
    results = harvest_survey(
        store,
        paper_id,
        limit=limit,
        section=section,
        summarize=summarize,
        language=_language(lang),
        progress=_progress,
    )
    if not results:
        console.print(
            "Nothing new to harvest: every arXiv reference is already in the library."
        )
        return
    _report(results)


@survey_app.command("merge")
def survey_merge(
    field_name: Optional[str] = typer.Option(None, "--field", help="Only this field"),
    papers: Optional[list[str]] = typer.Option(None, "--paper", "-p", help="Specific surveys"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Reconcile several surveys of a field into one map."""
    _setup_logging(verbose)
    store = _store()

    if papers:
        groups = {field_name or "": [_resolve_or_exit(store, item) for item in papers]}
    else:
        groups = {
            name: ids
            for name, ids in group_surveys_by_field(store).items()
            if not field_name or slugify(field_name) == slugify(name)
        }
    if not groups:
        console.print("No surveys with notes yet. Try `paper survey add <arxiv id>`.")
        return

    for name, survey_ids in groups.items():
        with console.status(f"Merging {len(survey_ids)} survey(s) for {name or 'field'}…"):
            try:
                path = merge_surveys(
                    store,
                    survey_ids,
                    field_name=name,
                    language=_language(lang),
                )
            except Exception as exc:
                console.print(f"[red]{name} failed:[/red] {exc}")
                continue
        console.print(f"[green]wrote[/green] {path.relative_to(store.settings.root)}")


@survey_app.command("core")
def survey_core(
    field_name: Optional[str] = typer.Option(None, "--field"),
    min_surveys: int = typer.Option(2, "--min-surveys"),
    limit: int = typer.Option(30, "--limit", "-n"),
    add: bool = typer.Option(False, "--add", help="Ingest the ones not in the library"),
    lang: Optional[str] = typer.Option(None, "--lang"),
) -> None:
    """Show references that several surveys agree on: the core of a field."""
    store = _store()
    grouped = group_surveys_by_field(store)
    if field_name:
        grouped = {
            name: ids for name, ids in grouped.items() if slugify(field_name) == slugify(name)
        }
    survey_ids = [paper_id for ids in grouped.values() for paper_id in ids]
    if len(survey_ids) < min_surveys:
        console.print(
            f"Need at least {min_surveys} surveys; the library has {len(survey_ids)}."
        )
        return

    shared = core_references(store, survey_ids, min_surveys=min_surveys)
    if not shared:
        console.print("No references are shared across that many surveys.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Surveys", justify="right")
    table.add_column("Cites", justify="right")
    table.add_column("arXiv")
    table.add_column("In lib")
    table.add_column("Reference", overflow="fold")
    for reference, count, _surveys in shared[:limit]:
        table.add_row(
            str(count),
            str(reference.citations),
            reference.arxiv_id or "-",
            "yes" if reference.in_library else "",
            (reference.title or reference.key)[:70],
        )
    console.print(table)

    if add:
        missing = [
            reference.arxiv_id
            for reference, _count, _surveys in shared[:limit]
            if reference.arxiv_id and not reference.in_library
        ]
        if not missing:
            console.print("\nAll of them are already in the library.")
            return
        console.print(f"\nIngesting {len(missing)} paper(s).")
        _report(
            ingest_inputs(
                store,
                missing,
                summarize=LLMClient(store.settings.llm).is_configured,
                language=_language(lang),
                kind="paper",
                progress=_progress,
            )
        )


@app.command()
def rebuild(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Re-run LaTeX conversion for every paper."""
    _setup_logging(verbose)
    store = _store()
    console.print(f"Reconverted {rebuild_all(store, progress=_progress)} paper(s).")


@app.command()
def summarize(
    query: Optional[str] = typer.Argument(None, help="Paper id or title fragment"),
    all_papers: bool = typer.Option(False, "--all", help="Redo notes that already exist too"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Write notes. With no argument, covers every paper that lacks one."""
    _setup_logging(verbose)
    store = _store()
    if not LLMClient(store.settings.llm).is_configured:
        console.print(
            f"[red]No API key.[/red] Set {store.settings.llm.api_key_env} in .env first."
        )
        raise typer.Exit(1)

    if query:
        targets = [_resolve_or_exit(store, query)]
    else:
        targets = [
            paper_id
            for paper_id in store.list_ids()
            if all_papers or store.load_summary(paper_id) is None
        ]
    if not targets:
        console.print("Every paper already has a note. Use --all to regenerate them.")
        return

    failures = 0
    for position, paper_id in enumerate(targets, start=1):
        console.print(f"[dim]({position}/{len(targets)})[/dim] {paper_id}")
        try:
            with console.status(f"Summarizing {paper_id}…"):
                summary = summarize_paper(store, paper_id, language=_language(lang))
        except Exception as exc:
            failures += 1
            console.print(f"  [red]failed:[/red] {exc}")
            continue
        console.print(f"  [green]{summary.one_liner or 'done'}[/green]")

    write_index(store)
    console.print(
        f"\n{len(targets) - failures}/{len(targets)} summarized. "
        "Next: `paper digest` to synthesize topics."
    )


# ---------------------------------------------------------------------------
# Browsing
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_papers(
    topic: Optional[str] = typer.Option(None, "--topic", help="Filter by topic"),
) -> None:
    """List everything in the library."""
    store = _store()
    records = list(store.iter_records())
    if topic:
        records = [
            record
            for record in records
            if record.summary
            and any(slugify(item) == slugify(topic) for item in record.summary.topics)
        ]
    if not records:
        console.print("Library is empty. Try `paper add 2503.07598`.")
        return

    records.sort(key=lambda record: record.meta.published or "", reverse=True)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Paper")
    table.add_column("Date")
    table.add_column("Title", overflow="fold")
    table.add_column("Topics", overflow="fold")
    table.add_column("Note")
    for record in records:
        table.add_row(
            record.meta.paper_id,
            (record.meta.published or "")[:10],
            record.meta.display_title,
            ", ".join(record.summary.topics) if record.summary else "",
            "yes" if record.summary else "[yellow]no[/yellow]",
        )
    console.print(table)
    console.print(f"\n{len(records)} paper(s).")


@app.command()
def show(
    query: str = typer.Argument(..., help="Paper id or title fragment"),
    raw: bool = typer.Option(False, "--raw", help="Print Markdown without rendering"),
) -> None:
    """Print a paper's note."""
    store = _store()
    paper_id = _resolve_or_exit(store, query)
    path = store.paper_dir(paper_id) / "summary.md"
    if not path.is_file():
        console.print(f"[yellow]{paper_id} has no note. Run `paper resummarize {paper_id}`.[/yellow]")
        raise typer.Exit(1)
    text = path.read_text(encoding="utf-8")
    console.print(text if raw else Markdown(text))


@app.command()
def info(query: str = typer.Argument(...)) -> None:
    """Show what the converter made of a paper's source tree."""
    store = _store()
    paper_id = _resolve_or_exit(store, query)
    stats = source_stats(store, paper_id)
    meta = store.load_meta(paper_id)
    table = Table(show_header=False)
    table.add_row("paper_id", paper_id)
    table.add_row("title", meta.display_title if meta else "")
    table.add_row("root tex", str(stats["root_tex"]))
    table.add_row("tex files", str(stats["tex_files"]))
    table.add_row("fulltext chars", str(stats["fulltext_chars"]))
    table.add_row("has note", "yes" if store.load_summary(paper_id) else "no")
    console.print(table)


@app.command()
def remove(
    query: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete a paper from the library."""
    store = _store()
    paper_id = _resolve_or_exit(store, query)
    if not yes:
        typer.confirm(f"Delete {paper_id} and all its files?", abort=True)
    console.print("Removed." if store.delete(paper_id) else "Nothing to remove.")
    write_index(store)


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------

@app.command()
def ask(
    question: list[str] = typer.Argument(..., help="Your question"),
    paper: Optional[list[str]] = typer.Option(None, "--paper", "-p", help="Restrict to papers"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    top_k: int = typer.Option(10, "--top-k", help="How many excerpts to retrieve"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ask a question about the library."""
    _setup_logging(verbose)
    store = _store()
    paper_ids = [_resolve_or_exit(store, item) for item in (paper or [])] or None
    text = " ".join(question)
    with console.status("Thinking…"):
        answer = ask_library(
            store, text, paper_ids=paper_ids, language=_language(lang), top_k=top_k
        )
    console.print(Markdown(answer.text))
    if answer.sources:
        console.print("\n[dim]Sources: " + "; ".join(answer.sources) + "[/dim]")


@app.command()
def compare(
    papers: list[str] = typer.Argument(..., help="Two or more paper ids"),
    aspect: str = typer.Option("", "--aspect", "-a", help="What to focus on"),
    lang: Optional[str] = typer.Option(None, "--lang"),
) -> None:
    """Compare several papers side by side."""
    store = _store()
    paper_ids = [_resolve_or_exit(store, item) for item in papers]
    if len(paper_ids) < 2:
        console.print("[red]Need at least two papers.[/red]")
        raise typer.Exit(1)
    with console.status("Comparing…"):
        text = compare_papers(store, paper_ids, aspect=aspect, language=_language(lang))
    console.print(Markdown(text))


# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------

@app.command()
def topics() -> None:
    """List the topics the library has been grouped into."""
    store = _store()
    grouped = topic_index(store)
    if not grouped:
        console.print("No topics yet. Summarize some papers first.")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Topic")
    table.add_column("Papers", justify="right")
    table.add_column("Digest")
    for topic, records in grouped.items():
        path = store.settings.knowledge_dir / "topics" / f"{slugify(topic)}.md"
        table.add_row(topic, str(len(records)), "yes" if path.is_file() else "-")
    console.print(table)


@app.command()
def digest(
    topic: Optional[str] = typer.Option(None, "--topic", help="Only this topic"),
    min_papers: int = typer.Option(2, "--min-papers"),
    lang: Optional[str] = typer.Option(None, "--lang"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Write cross-paper syntheses into knowledge/topics/."""
    _setup_logging(verbose)
    store = _store()
    with console.status("Synthesizing…"):
        written = write_all_topic_digests(
            store, min_papers=min_papers, language=_language(lang), only=topic
        )
    if not written:
        console.print(
            "Nothing written. Topics need at least "
            f"{min_papers} papers, or pass --topic to force one."
        )
        return
    for path in written:
        console.print(f"[green]wrote[/green] {path.relative_to(store.settings.root)}")


@app.command()
def overview(
    lang: Optional[str] = typer.Option(None, "--lang"),
    show_it: bool = typer.Option(True, "--show/--no-show"),
) -> None:
    """Write the library-wide overview page."""
    store = _store()
    with console.status("Synthesizing…"):
        path = write_overview(store, language=_language(lang))
    console.print(f"[green]wrote[/green] {path.relative_to(store.settings.root)}")
    if show_it:
        console.print(Markdown(path.read_text(encoding="utf-8")))


@app.command()
def glossary() -> None:
    """Write the concept glossary."""
    store = _store()
    path = write_glossary(store)
    console.print(f"[green]wrote[/green] {path.relative_to(store.settings.root)}")


@app.command()
def index() -> None:
    """Refresh knowledge/index.md."""
    store = _store()
    path = write_index(store)
    console.print(f"[green]wrote[/green] {path.relative_to(store.settings.root)}")


# ---------------------------------------------------------------------------
# Interactive and server
# ---------------------------------------------------------------------------

@app.command()
def chat() -> None:
    """Talk to the same command router the chat bots use."""
    from .bots.router import BotContext, Router

    store = _store()
    router = Router(store)
    context = BotContext(platform="cli", user_id="local", chat_id="local")
    console.print("[bold]surveyor[/bold] — type `help`, or `exit` to leave.\n")
    while True:
        try:
            message = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return
        if message.lower() in {"exit", "quit", ":q"}:
            return
        if not message:
            continue
        reply = router.handle(message, context)
        if reply.ack:
            console.print(Markdown(reply.ack))
        if reply.work:
            with console.status("Working…"):
                result = reply.work()
            if result:
                console.print(Markdown(result))


@app.command()
def gui(
    port: int = typer.Option(8760, "--port", "-p", help="Port on localhost"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not open a browser"),
) -> None:
    """Open the Surveyor app in your browser."""
    from .gui import run

    console.print(f"[bold]Surveyor[/bold] is starting on http://127.0.0.1:{port}")
    console.print("Press Ctrl+C to stop.\n")
    run(port=port, open_browser=not no_browser)


@app.command()
def home(
    path: Optional[Path] = typer.Argument(None, help="Folder to keep papers and notes in"),
    forget: bool = typer.Option(False, "--forget", help="Go back to the default folder"),
) -> None:
    """Show or change where the library lives."""
    from .config import reload_settings
    from .userconfig import forget_library_root, set_library_root

    if forget:
        forget_library_root()
        console.print("Reset. The library folder is now chosen automatically.")
    elif path is not None:
        chosen = set_library_root(path)
        console.print(f"Library folder set to [bold]{chosen}[/bold]")
    console.print(f"Currently using [bold]{reload_settings().root}[/bold]")


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the webhook server for the Feishu and WeCom bots."""
    import uvicorn

    from .bots.server import create_app

    settings = get_settings()
    host = host or settings.server.host
    port = port or settings.server.port
    console.print(f"Serving on http://{host}:{port}")
    console.print("  Feishu webhook: POST /webhook/feishu")
    console.print("  WeCom webhook:  GET+POST /webhook/wecom")
    uvicorn.run(create_app(settings), host=host, port=port, reload=reload)


@app.command(name="feishu-connect")
def feishu_connect(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run the Feishu bot over a long connection, with no public URL needed."""
    from .bots.feishu_longconn import run

    _setup_logging(verbose, default=logging.INFO)
    console.print("Connecting to Feishu over a long connection.")
    console.print("  No public URL, request URL or encryption key is needed.")
    console.print("  Press Ctrl+C to stop.\n")
    try:
        run(get_settings())
    except RuntimeError as exc:
        # These messages name TOML tables like [feishu], which Rich would read as
        # markup and swallow.
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\nDisconnected.")


@app.command(name="wecom-connect")
def wecom_connect(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Run the WeCom 智能机器人 over a long connection, with no public URL needed."""
    from .bots.wecom_longconn import run

    _setup_logging(verbose, default=logging.INFO)
    console.print("Connecting to WeCom over a long connection.")
    console.print("  This is the 智能机器人 route: it needs WECOM_BOT_ID and")
    console.print("  WECOM_BOT_SECRET, not the self-built app's token and AES key.")
    console.print("  Press Ctrl+C to stop.\n")
    try:
        run(get_settings())
    except RuntimeError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\nDisconnected.")


@app.command()
def status() -> None:
    """Show configuration and library health."""
    settings = get_settings()
    store = PaperStore(settings)
    records = list(store.iter_records())
    client = LLMClient(settings.llm)

    table = Table(show_header=False)
    table.add_row("root", str(settings.root))
    table.add_row("papers", f"{len(records)} ({sum(1 for r in records if r.summary)} with notes)")
    survey_ids = store.list_surveys()
    if survey_ids:
        harvestable = sum(
            1
            for paper_id in survey_ids
            for reference in store.load_references(paper_id)
            if reference.arxiv_id and not reference.in_library
        )
        table.add_row(
            "surveys",
            f"{len(survey_ids)} ({harvestable} cited arXiv papers not yet ingested)",
        )
    table.add_row("topics", str(len(topic_index(store))))
    table.add_row("model", settings.llm.model)
    table.add_row("base url", settings.llm.base_url)
    table.add_row(
        "api key",
        "[green]configured[/green]" if client.is_configured
        else f"[red]missing[/red] (set {settings.llm.api_key_env})",
    )
    table.add_row("language", settings.default_language.value)
    table.add_row(
        "feishu",
        "[green]on[/green]" if settings.feishu.enabled and settings.feishu.app_id
        else "off",
    )
    # The two WeCom routes take different credentials and cannot both be live, so
    # naming the configured one saves a round of guessing.
    wecom = settings.wecom
    if wecom.enabled and wecom.bot_id:
        wecom_state = "[green]on[/green] (long connection)"
    elif wecom.enabled and wecom.corp_id:
        wecom_state = "[green]on[/green] (self-built app)"
    else:
        wecom_state = "off"
    table.add_row("wecom", wecom_state)
    console.print(table)


if __name__ == "__main__":
    app()
