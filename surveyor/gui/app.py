"""The local web app.

A FastAPI server that exposes the same operations as the command line over
HTTP, plus a single-page UI in ``static/``. It listens on the loopback
interface only: it can spend money on API calls and write to the library, so it
is never something to expose on a network.
"""

from __future__ import annotations

import logging
import socket
import threading
import webbrowser
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..bots.imaging import available as image_rendering_available
from ..config import (
    Language,
    Settings,
    default_library_root,
    get_settings,
    library_root,
    reload_settings,
)
from ..knowledge import topic_index, write_all_topic_digests, write_glossary, write_overview
from ..llm import LLMClient, LLMError
from ..models import SurveyReference
from ..paths import knowledge_root, slugify
from ..pipeline import harvest_survey, ingest_inputs, rebuild_all
from ..qa import ask as ask_library
from ..qa import compare as compare_papers
from ..render import to_html
from ..store import PaperStore, write_all_indexes, write_index
from ..summarize import summarize_paper
from ..survey import (
    collect_references,
    core_references,
    group_surveys_by_field,
    merge_surveys,
    reading_list,
    resolve_missing_ids,
    summarize_survey,
)
from ..userconfig import apply as apply_settings
from ..userconfig import read_env_file, set_library_root
from .jobs import JobRunner

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Browsers will not send a custom header to another origin without first making
# a preflight request, which this server answers for nobody. That makes the
# header a sufficient guard against a web page in another tab driving the app.
GUARD_HEADER = "x-surveyor-app"


def _store() -> PaperStore:
    return PaperStore(get_settings())


def _language(value: str | None) -> Language | None:
    return Language.parse(value, get_settings().default_language) if value else None


def _error(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


# ---------------------------------------------------------------- serializers


def _paper_row(store: PaperStore, paper_id: str) -> dict[str, Any] | None:
    meta = store.load_meta(paper_id)
    if meta is None:
        return None
    summary = store.load_summary(paper_id)
    return {
        "paper_id": meta.paper_id,
        "arxiv_id": meta.arxiv_id,
        "title": meta.display_title,
        "authors": meta.author_line,
        "published": (meta.published or "")[:10],
        "kind": meta.kind,
        "collection": meta.collection,
        "has_source": meta.has_source,
        "has_note": summary is not None,
        "one_liner": summary.one_liner if summary else "",
        "topics": summary.topics if summary else [],
        "abs_url": meta.abs_url,
    }


def _reference_row(reference: SurveyReference) -> dict[str, Any]:
    return {
        "key": reference.key,
        "title": reference.title,
        "year": reference.year,
        "arxiv_id": reference.arxiv_id,
        "citations": reference.citations,
        "table_citations": reference.table_citations,
        "section": reference.primary_section,
        "in_library": reference.in_library,
    }


def _settings_payload(settings: Settings) -> dict[str, Any]:
    env = read_env_file(settings.root)
    return {
        "root": str(settings.root),
        "default_root": str(default_library_root()),
        "default_language": settings.default_language.value,
        "llm": {
            "base_url": settings.llm.base_url,
            "model": settings.llm.model,
            "fast_model": settings.llm.fast_model or "",
            "temperature": settings.llm.temperature,
            "timeout": settings.llm.timeout,
            "max_context_chars": settings.llm.max_context_chars,
            "api_key_env": settings.llm.api_key_env,
            "has_key": bool(settings.llm.api_key),
        },
        "arxiv": {
            "user_agent": settings.arxiv.user_agent,
            "request_interval": settings.arxiv.request_interval,
        },
        "bots": {
            "feishu_enabled": settings.feishu.enabled,
            "wecom_enabled": settings.wecom.enabled,
            "send_as_image": settings.feishu.send_as_image,
            "can_draw": image_rendering_available(),
            "port": settings.server.port,
            "feishu_app_id": env.get("FEISHU_APP_ID", ""),
            "has_feishu_secret": bool(env.get("FEISHU_APP_SECRET")),
            "has_wecom_token": bool(env.get("WECOM_TOKEN")),
            # Which WeCom route is set up decides which command to run, so the
            # page has to be able to tell them apart.
            "has_wecom_bot": bool(env.get("WECOM_BOT_ID")),
        },
    }


def _library_stats(store: PaperStore) -> dict[str, Any]:
    papers = surveys = notes = 0
    for record in store.iter_records():
        papers += 1
        if record.meta.kind == "survey":
            surveys += 1
        if record.summary:
            notes += 1
    harvestable = 0
    for survey_id in store.list_surveys():
        harvestable += sum(
            1
            for reference in store.load_references(survey_id)
            if reference.arxiv_id and not reference.in_library
        )
    return {
        "papers": papers - surveys,
        "surveys": surveys,
        "notes": notes,
        "harvestable": harvestable,
        "root": str(store.settings.root),
        "root_display": _shorten(store.settings.root),
    }


def _shorten(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------- routes


def create_app() -> FastAPI:
    app = FastAPI(title="Surveyor", docs_url=None, redoc_url=None)
    runner = JobRunner()

    @app.middleware("http")
    async def guard(request: Request, call_next):  # noqa: ANN001 - FastAPI signature
        if request.url.path.startswith("/api/") and request.headers.get(GUARD_HEADER) != "1":
            return JSONResponse(
                {"detail": "This endpoint is only callable from the Surveyor app."},
                status_code=403,
            )
        return await call_next(request)

    # ------------------------------------------------------------ state
    @app.get("/api/state")
    def state() -> dict[str, Any]:
        settings = get_settings()
        store = PaperStore(settings)
        return {
            "library": _library_stats(store),
            "settings": _settings_payload(settings),
            "configured": bool(settings.llm.api_key),
            "busy": runner.busy,
        }

    @app.post("/api/settings")
    def save_settings(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        root = Path(payload.get("root") or library_root()).expanduser()
        try:
            set_library_root(root)
        except OSError as exc:
            raise _error(400, f"Cannot use that folder: {exc}") from exc

        secrets = {name: value for name, value in (payload.get("secrets") or {}).items()}
        config = {
            "default_language": payload.get("default_language", "zh"),
            "llm": payload.get("llm", {}),
            "arxiv": payload.get("arxiv", {}),
            "server": payload.get("server", {}),
            "feishu": payload.get("feishu", {}),
            "wecom": payload.get("wecom", {}),
        }
        settings = apply_settings(Path(root).expanduser().resolve(), config, secrets)
        settings.ensure_dirs()
        return {"settings": _settings_payload(settings), "library": _library_stats(PaperStore(settings))}

    @app.post("/api/settings/test")
    def test_llm(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Send the smallest possible completion to prove the endpoint works."""
        settings = get_settings()
        config = settings.llm.model_copy(
            update={
                "base_url": payload.get("base_url") or settings.llm.base_url,
                "model": payload.get("model") or settings.llm.model,
                "max_retries": 1,
                "timeout": 30.0,
            }
        )
        client = LLMClient(config)
        if not client.is_configured:
            return {"ok": False, "message": f"No API key in {config.api_key_env}."}
        try:
            reply = client.complete(
                [{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=16,
                temperature=0,
            )
        except LLMError as exc:
            return {"ok": False, "message": str(exc)[:400]}
        except Exception as exc:
            return {"ok": False, "message": f"{type(exc).__name__}: {exc}"[:400]}
        return {"ok": True, "message": f"{config.model} replied: {reply[:80]}"}

    # ---------------------------------------------------------- library
    @app.get("/api/papers")
    def papers(
        kind: str = Query("all"),
        q: str = Query(""),
        collection: str | None = Query(None),
    ) -> dict[str, Any]:
        store = _store()
        rows = [row for pid in store.list_ids() if (row := _paper_row(store, pid))]
        if kind in ("paper", "survey"):
            rows = [row for row in rows if row["kind"] == kind]
        if collection is not None:
            rows = [row for row in rows if row["collection"] == collection]
        if q:
            needle = q.lower()
            rows = [
                row
                for row in rows
                if needle in row["title"].lower()
                or needle in row["paper_id"].lower()
                or needle in row["one_liner"].lower()
            ]
        rows.sort(key=lambda row: (row["published"] or ""), reverse=True)
        return {"papers": rows}

    # ------------------------------------------------------ collections
    @app.get("/api/collections")
    def collections() -> dict[str, Any]:
        store = _store()
        return {
            "collections": [
                {"name": name, "papers": count} for name, count in store.list_collections()
            ],
            "unfiled": sum(1 for _ in store.iter_records("")),
        }

    @app.post("/api/collections/assign")
    def assign_collection(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        store = _store()
        paper_ids = payload.get("papers") or []
        if not paper_ids:
            raise _error(400, "Select the papers to file first.")
        name = (payload.get("collection") or "").strip()
        moved = [pid for pid in paper_ids if store.set_collection(pid, name)]
        if moved:
            write_all_indexes(store)
        return {"moved": moved, "collection": name}

    @app.get("/api/papers/{paper_id}")
    def paper_detail(paper_id: str) -> dict[str, Any]:
        store = _store()
        meta = store.load_meta(paper_id)
        if meta is None:
            raise _error(404, f"{paper_id} is not in the library.")
        directory = store.paper_dir(paper_id)
        note_file = "survey.md" if meta.kind == "survey" else "summary.md"
        note_path = directory / note_file
        note = note_path.read_text(encoding="utf-8") if note_path.is_file() else ""
        return {
            "paper": _paper_row(store, paper_id),
            "abstract": meta.abstract,
            "note_markdown": note,
            "note_html": to_html(note),
            "fulltext_chars": len(store.load_fulltext(paper_id)),
            "folder": str(directory),
        }

    @app.delete("/api/papers/{paper_id}")
    def remove_paper(paper_id: str) -> dict[str, Any]:
        store = _store()
        if not store.delete(paper_id):
            raise _error(404, f"{paper_id} is not in the library.")
        write_index(store)
        return {"removed": paper_id}

    @app.post("/api/papers/{paper_id}/summarize")
    def summarize_one(paper_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        store = _store()
        meta = store.load_meta(paper_id)
        if meta is None:
            raise _error(404, f"{paper_id} is not in the library.")
        language = _language(payload.get("language"))

        def work(progress):
            progress(f"{paper_id}: reading")
            if meta.kind == "survey":
                note = summarize_survey(store, paper_id, language=language)
                progress(f"{paper_id}: mapped {len(note.taxonomy)} top-level branches")
            else:
                summary = summarize_paper(store, paper_id, language=language)
                progress(f"{paper_id}: {summary.one_liner[:80]}")
            write_index(store)
            return {"paper_id": paper_id}

        return _job(runner, f"Summarize {paper_id}", work)

    @app.post("/api/ingest")
    def ingest(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        raw = (payload.get("text") or "").strip()
        if not raw:
            raise _error(400, "Paste at least one arXiv id, URL or title.")
        inputs = [line.strip() for line in raw.splitlines() if line.strip()]
        kind = payload.get("kind") or None
        summarize = bool(payload.get("summarize", True))
        language = _language(payload.get("language"))
        collection = (payload.get("collection") or "").strip()
        label = f"Import {len(inputs)} item(s)"

        def work(progress):
            store = _store()
            results = ingest_inputs(
                store,
                inputs,
                summarize=summarize,
                language=language,
                kind=kind if kind in ("paper", "survey") else None,
                progress=progress,
            )
            if collection:
                for result in results:
                    if result.paper_id:
                        store.set_collection(result.paper_id, collection)
                write_all_indexes(store)
            return [
                {
                    "paper_id": result.paper_id,
                    "status": result.status(),
                    "title": result.title,
                    "is_survey": result.is_survey,
                    "errors": result.errors,
                }
                for result in results
            ]

        return _job(runner, label, work)

    @app.post("/api/rebuild")
    def rebuild() -> dict[str, Any]:
        def work(progress):
            return {"converted": rebuild_all(_store(), progress=progress)}

        return _job(runner, "Reconvert every LaTeX source", work)

    # ----------------------------------------------------------- surveys
    @app.get("/api/surveys")
    def surveys() -> dict[str, Any]:
        store = _store()
        rows = []
        for paper_id in store.list_surveys():
            row = _paper_row(store, paper_id)
            if row is None:
                continue
            references = store.load_references(paper_id)
            note = store.load_survey(paper_id)
            row |= {
                "references": len(references),
                "on_arxiv": sum(1 for reference in references if reference.arxiv_id),
                "harvested": sum(1 for reference in references if reference.in_library),
                "field_name": note.field_name if note else "",
                "branches": len(note.taxonomy) if note else 0,
            }
            rows.append(row)
        return {"surveys": rows, "fields": group_surveys_by_field(store)}

    @app.get("/api/surveys/{paper_id}")
    def survey_detail(paper_id: str) -> dict[str, Any]:
        store = _store()
        meta = store.load_meta(paper_id)
        if meta is None:
            raise _error(404, f"{paper_id} is not in the library.")
        note = store.load_survey(paper_id)
        taxonomy = (
            [
                {"depth": depth, "name": node.name, "description": node.description}
                for depth, node in note.flat_taxonomy()
            ]
            if note
            else []
        )
        note_path = store.paper_dir(paper_id) / "survey.md"
        markdown = note_path.read_text(encoding="utf-8") if note_path.is_file() else ""
        return {
            "paper": _paper_row(store, paper_id),
            "field_name": note.field_name if note else "",
            "scope": note.scope if note else "",
            "taxonomy": taxonomy,
            "reading_path": note.reading_path if note else [],
            "open_challenges": note.open_challenges if note else [],
            "note_html": to_html(markdown),
            "has_note": note is not None,
        }

    @app.get("/api/surveys/{paper_id}/references")
    def survey_references(
        paper_id: str,
        limit: int = Query(60, ge=1, le=1000),
        section: str = Query(""),
        missing_only: bool = Query(False),
    ) -> dict[str, Any]:
        store = _store()
        references = store.load_references(paper_id)
        if not references and store.exists(paper_id):
            references = collect_references(store, paper_id)
        picks = reading_list(
            references,
            limit=limit,
            section=section or None,
            only_arxiv=False,
            only_missing=missing_only,
        )
        sections = sorted(
            {reference.primary_section for reference in references if reference.primary_section}
        )
        return {
            "total": len(references),
            "on_arxiv": sum(1 for reference in references if reference.arxiv_id),
            "in_library": sum(1 for reference in references if reference.in_library),
            "sections": sections,
            "references": [_reference_row(reference) for reference in picks],
        }

    @app.post("/api/surveys/{paper_id}/harvest")
    def harvest(paper_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        limit = int(payload.get("limit", 10))
        section = payload.get("section") or None
        summarize = bool(payload.get("summarize", True))
        language = _language(payload.get("language"))

        def work(progress):
            results = harvest_survey(
                _store(),
                paper_id,
                limit=limit,
                section=section,
                summarize=summarize,
                language=language,
                progress=progress,
            )
            return [
                {"paper_id": result.paper_id, "status": result.status(), "title": result.title}
                for result in results
            ]

        return _job(runner, f"Harvest {limit} paper(s) from {paper_id}", work)

    @app.post("/api/surveys/{paper_id}/resolve")
    def resolve_ids(paper_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        limit = int(payload.get("limit", 30))

        def work(progress):
            found = resolve_missing_ids(_store(), paper_id, limit=limit, progress=progress)
            progress(f"resolved {found} arXiv id(s)")
            return {"found": found}

        return _job(runner, f"Look up arXiv ids for {paper_id}", work)

    @app.post("/api/surveys/{paper_id}/references/refresh")
    def refresh_references(paper_id: str) -> dict[str, Any]:
        def work(progress):
            references = collect_references(_store(), paper_id)
            progress(f"{len(references)} cited references")
            return {"references": len(references)}

        return _job(runner, f"Re-extract references from {paper_id}", work)

    @app.get("/api/core")
    def core(min_surveys: int = Query(2, ge=1), field: str = Query("")) -> dict[str, Any]:
        store = _store()
        grouped = group_surveys_by_field(store)
        survey_ids = grouped.get(field, []) if field else store.list_surveys()
        shared = core_references(store, survey_ids, min_surveys=min_surveys)
        return {
            "surveys": survey_ids,
            "references": [
                _reference_row(reference) | {"agreeing": count, "cited_by": ids}
                for reference, count, ids in shared
            ],
        }

    @app.post("/api/surveys/merge")
    def merge(payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        field = payload.get("field") or ""
        language = _language(payload.get("language"))
        collection = (payload.get("collection") or "").strip() or None

        def work(progress):
            store = _store()
            grouped = group_surveys_by_field(store)
            if collection is not None:
                filed = {
                    record.meta.paper_id for record in store.iter_records(collection)
                }
                grouped = {
                    name: [pid for pid in ids if pid in filed]
                    for name, ids in grouped.items()
                }
            targets = {field: grouped.get(field, [])} if field else grouped
            written = []
            for name, ids in targets.items():
                if not ids:
                    continue
                progress(f"{name}: reconciling {len(ids)} survey(s)")
                path = merge_surveys(
                    store, ids, field_name=name, language=language, collection=collection
                )
                written.append(str(path))
            if not written:
                raise ValueError("No surveys with notes to merge yet.")
            return {"written": written}

        return _job(runner, "Reconcile surveys into a field map", work)

    # --------------------------------------------------------------- ask
    @app.post("/api/ask")
    def ask(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        question = (payload.get("question") or "").strip()
        if not question:
            raise _error(400, "Ask something first.")
        paper_ids = payload.get("papers") or None
        language = _language(payload.get("language"))

        def work(progress):
            progress("searching the library")
            answer = ask_library(
                _store(), question, paper_ids=paper_ids, language=language
            )
            return {
                "question": question,
                "html": to_html(answer.text),
                "sources": answer.sources,
                "papers": answer.used_papers,
            }

        return _job(runner, f"Ask: {question[:60]}", work)

    @app.post("/api/compare")
    def compare(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        paper_ids = payload.get("papers") or []
        if len(paper_ids) < 2:
            raise _error(400, "Pick at least two papers to compare.")
        aspect = payload.get("aspect") or ""
        language = _language(payload.get("language"))

        def work(progress):
            progress(f"comparing {len(paper_ids)} papers")
            text = compare_papers(
                _store(), paper_ids, aspect=aspect, language=language
            )
            return {"html": to_html(text), "papers": paper_ids}

        return _job(runner, f"Compare {len(paper_ids)} papers", work)

    # --------------------------------------------------------- knowledge
    @app.get("/api/knowledge")
    def knowledge(collection: str | None = Query(None)) -> dict[str, Any]:
        store = _store()
        root = store.settings.knowledge_dir
        # Documents belonging to a collection live one level down. Listing the
        # library therefore means everything at the top level only, so a
        # sub-library's pages do not show up twice.
        scoped = knowledge_root(store.settings, collection)
        folders = {slugify(name) for name, _count in store.list_collections()}
        documents = [
            {
                "name": str(path.relative_to(root)),
                "folder": str(path.parent.relative_to(scoped)) if path.parent != scoped else "",
                "title": _first_heading(path) or path.stem,
                "modified": int(path.stat().st_mtime),
            }
            for path in sorted(scoped.rglob("*.md"))
            if collection or path.relative_to(root).parts[0] not in folders
        ]
        return {
            "documents": documents,
            "collection": (collection or ""),
            "topics": [
                {"name": topic, "papers": len(records)}
                for topic, records in topic_index(store, collection).items()
            ],
        }

    @app.get("/api/knowledge/doc")
    def knowledge_doc(name: str = Query(...)) -> dict[str, Any]:
        store = _store()
        directory = store.settings.knowledge_dir.resolve()
        path = (directory / name).resolve()
        if not path.is_file() or directory not in path.parents:
            raise _error(404, f"No such document: {name}")
        text = path.read_text(encoding="utf-8")
        return {"name": name, "html": to_html(text), "markdown": text}

    @app.post("/api/knowledge/build")
    def build_knowledge(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        what = payload.get("what", "overview")
        language = _language(payload.get("language"))
        topic = payload.get("topic") or None
        collection = (payload.get("collection") or "").strip() or None

        def work(progress):
            store = _store()
            if what == "glossary":
                progress("collecting concepts from every note")
                return {"written": [str(write_glossary(store, collection))]}
            if what == "index":
                paths = [write_index(store, collection)] if collection \
                    else write_all_indexes(store)
                return {"written": [str(path) for path in paths]}
            if what == "digests":
                progress("synthesizing topic digests")
                paths = write_all_topic_digests(
                    store, min_papers=1 if topic else 2, language=language, only=topic,
                    collection=collection,
                )
                if not paths:
                    raise ValueError("No topic has enough summarized papers yet.")
                return {"written": [str(path) for path in paths]}
            progress("writing the overview")
            return {
                "written": [
                    str(write_overview(store, language=language, collection=collection))
                ]
            }

        return _job(runner, f"Build {what}", work)

    # -------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return {"jobs": [job.as_dict() for job in runner.recent()]}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, since: int = Query(0, ge=0)) -> dict[str, Any]:
        job = runner.get(job_id)
        if job is None:
            raise _error(404, "That task is no longer in the log.")
        return job.as_dict(since=since)

    # ------------------------------------------------------------ static
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _job(runner: JobRunner, label: str, work) -> dict[str, Any]:
    job = runner.start(label, work)
    return {"id": job.id, "label": job.label}


def _first_heading(path: Path) -> str:
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("# "):
                    return line[2:].strip()
    except OSError:
        pass
    return ""


# --------------------------------------------------------------------- launch


def _free_port(preferred: int) -> int:
    """Use the preferred port when it is free, otherwise let the OS pick one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run(port: int = 8760, open_browser: bool = True) -> None:
    import uvicorn

    reload_settings()
    port = _free_port(port)
    url = f"http://127.0.0.1:{port}"

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")


__all__ = ["create_app", "run"]
