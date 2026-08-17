"use strict";

/* --------------------------------------------------------------- plumbing */

async function api(path, options = {}) {
  const response = await fetch("/api" + path, {
    method: options.method || (options.body ? "POST" : "GET"),
    headers: { "Content-Type": "application/json", "X-Surveyor-App": "1" },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function toast(message, bad = false) {
  const node = document.createElement("div");
  if (bad) node.className = "bad";
  node.textContent = message;
  $("toast").append(node);
  setTimeout(() => node.remove(), bad ? 9000 : 4500);
}

/* ------------------------------------------------------------------ jobs */

const activity = {
  log: () => $("activity-log"),
  write(line) {
    const element = this.log();
    element.textContent += (element.textContent ? "\n" : "") + line;
    element.scrollTop = element.scrollHeight;
  },
  set(label, status) {
    $("activity-label").textContent = label;
    $("activity-dot").className = "dot" + (status ? " " + status : "");
  },
};

async function runJob(request, onDone) {
  let job;
  try {
    job = await request;
  } catch (error) {
    toast(error.message, true);
    return;
  }
  activity.set(job.label, "busy");
  activity.write("— " + job.label);
  $("activity").open = true;

  let seen = 0;
  const poll = async () => {
    let status;
    try {
      status = await api(`/jobs/${job.id}?since=${seen}`);
    } catch (error) {
      activity.set("Lost track of that task", "failed");
      return;
    }
    status.lines.forEach((line) => activity.write("  " + line));
    seen = status.total_lines;

    if (status.status === "running") {
      setTimeout(poll, 900);
      return;
    }
    activity.set(status.status === "done" ? "Idle" : "Last task failed",
                 status.status === "done" ? "" : "failed");
    if (status.status === "failed") {
      toast(status.error || "That task failed. See the log.", true);
    }
    await refreshState();
    if (onDone) onDone(status);
  };
  setTimeout(poll, 400);
}

/* ----------------------------------------------------------------- state */

const state = {
  settings: null, stats: null, papers: [], collections: [], view: "library",
};

async function refreshState() {
  const data = await api("/state");
  state.settings = data.settings;
  state.stats = data.library;
  $("library-root").textContent = data.library.root_display;
  $("library-root").title = data.library.root;
  $("count-papers").textContent = data.library.papers + data.library.surveys || "";
  $("count-surveys").textContent = data.library.surveys || "";
  return data;
}

const TOP_LEVEL = ["library", "surveys", "canon", "ask", "knowledge", "settings"];

/* Views live in the address bar so the back button works and a paper can be
   bookmarked. `hash` tracks what we put there, to tell our own navigation apart
   from the user pressing back. */
let hash = "";

function setHash(value) {
  hash = "#" + value;
  if (location.hash !== hash) location.hash = value;
}

function show(view, remember = true) {
  state.view = view;
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.id === "view-" + view);
  });
  document.querySelectorAll("#nav button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelector("main").scrollTop = 0;
  if (remember && TOP_LEVEL.includes(view)) setHash(view);
  const loader = LOADERS[view];
  if (loader) loader();
}

function route(raw) {
  hash = raw;
  const [view, id] = raw.replace(/^#/, "").split("/");
  const target = id ? decodeURIComponent(id) : "";
  if (view === "paper" && target) return openPaper(target, false);
  if (view === "survey" && target) return openSurvey(target, false);
  if (view === "doc" && target) return openDoc(target, false);
  show(TOP_LEVEL.includes(view) ? view : "library", false);
}

window.addEventListener("hashchange", () => {
  if (location.hash !== hash) route(location.hash);
});

const LOADERS = {
  library: loadLibrary,
  surveys: loadSurveys,
  canon: loadCanon,
  knowledge: loadKnowledge,
  settings: fillSettings,
  ask: async () => {
    if (!state.papers.length) {
      const data = await api("/papers?kind=all&q=");
      state.papers = data.papers;
      fillScopeSelect(data.papers);
    }
  },
};

/* --------------------------------------------------------------- library */

async function loadLibrary() {
  const kind = $("library-kind").value;
  const query = $("library-search").value.trim();
  const chosen = $("library-collection").value;
  // An empty value means every collection; UNFILED asks for the papers with none.
  const scope = chosen === UNFILED ? "&collection=" :
    chosen ? `&collection=${encodeURIComponent(chosen)}` : "";
  const data = await api(`/papers?kind=${kind}&q=${encodeURIComponent(query)}${scope}`);
  state.papers = data.papers;
  fillScopeSelect(data.papers);
  filePicker?.setPapers(data.papers);
  await loadCollections();

  const list = $("library-list");
  if (!data.papers.length) {
    list.innerHTML = `<div class="empty">${
      query ? "Nothing matches that filter." : "Nothing imported yet. Paste an arXiv id above to begin."
    }</div>`;
    return;
  }
  list.innerHTML = data.papers.map((paper) => `
    <button class="item" data-paper="${esc(paper.paper_id)}">
      <div class="title">${esc(paper.title)}
        ${paper.kind === "survey" ? '<span class="tag">survey</span>' : ""}
        ${paper.collection ? `<span class="tag plain">${esc(paper.collection)}</span>` : ""}
        ${paper.has_note ? "" : '<span class="tag plain">no note</span>'}
      </div>
      <div class="meta">${esc(paper.paper_id)} · ${esc(paper.published || "date unknown")} · ${esc(paper.authors)}</div>
      ${paper.one_liner ? `<div class="meta">${esc(paper.one_liner)}</div>` : ""}
    </button>`).join("");

  list.querySelectorAll("[data-paper]").forEach((node) => {
    node.onclick = () => openPaper(node.dataset.paper);
  });
}

async function openPaper(paperId, remember = true) {
  let data;
  try {
    data = await api(`/papers/${encodeURIComponent(paperId)}`);
  } catch (error) {
    toast(error.message, true);
    return show("library");
  }
  const paper = data.paper;
  if (paper.kind === "survey") return openSurvey(paperId, remember);

  show("paper", false);
  if (remember) setHash("paper/" + encodeURIComponent(paperId));
  $("paper-title").textContent = paper.title;
  $("paper-meta").textContent =
    `${paper.paper_id} · ${paper.published || "date unknown"} · ${paper.authors}`;
  // Without a note the abstract is the next best thing, but it is the paper's
  // own words and must not read as a continuation of our notice.
  $("paper-note").innerHTML = data.note_html ||
    `<p class="muted">No note yet — the abstract, from arXiv:</p>
     <blockquote>${esc(data.abstract)}</blockquote>`;

  const link = $("paper-arxiv");
  link.href = paper.abs_url || `https://arxiv.org/abs/${paper.arxiv_id}`;
  link.classList.toggle("hidden", !paper.arxiv_id && !paper.abs_url);

  $("paper-summarize").textContent = paper.has_note ? "Rewrite the note" : "Write the note";
  $("paper-summarize").onclick = () =>
    runJob(api(`/papers/${encodeURIComponent(paperId)}/summarize`, { body: {} }),
           () => openPaper(paperId));
  $("paper-ask").onclick = () => {
    show("ask");
    askPicker.setSelection([paperId]);
    $("ask-question").focus();
  };
  $("paper-remove").onclick = async () => {
    if (!confirm(`Remove ${paperId} and everything under it?`)) return;
    await api(`/papers/${encodeURIComponent(paperId)}`, { method: "DELETE" });
    toast(`Removed ${paperId}`);
    await refreshState();
    show("library");
  };
}

/* --------------------------------------------------------------- surveys */

async function loadSurveys() {
  const data = await api("/surveys");
  const list = $("surveys-list");
  if (!data.surveys.length) {
    list.innerHTML = `<div class="empty">No surveys yet. Import one from the Library tab —
      choose "Survey" if the title does not say so.</div>`;
    return;
  }
  list.innerHTML = data.surveys.map((survey) => `
    <button class="item" data-survey="${esc(survey.paper_id)}">
      <div class="title">${esc(survey.title)}</div>
      <div class="meta">${esc(survey.paper_id)} · ${esc(survey.published || "date unknown")}</div>
      <div class="meta">${survey.references} references, ${survey.on_arxiv} on arXiv,
        ${survey.harvested} already imported ·
        ${survey.branches ? esc(survey.field_name || "mapped") : "taxonomy not mapped yet"}</div>
    </button>`).join("");
  list.querySelectorAll("[data-survey]").forEach((node) => {
    node.onclick = () => openSurvey(node.dataset.survey);
  });
}

let currentSurvey = null;

async function openSurvey(paperId, remember = true) {
  let data;
  try {
    data = await api(`/surveys/${encodeURIComponent(paperId)}`);
  } catch (error) {
    toast(error.message, true);
    return show("surveys");
  }
  currentSurvey = paperId;
  show("survey", false);
  if (remember) setHash("survey/" + encodeURIComponent(paperId));

  $("survey-title").textContent = data.paper.title;
  $("survey-meta").textContent =
    `${data.paper.paper_id} · ${data.field_name || "field not named yet"} · ${data.paper.authors}`;

  $("survey-taxonomy").innerHTML = data.taxonomy.length
    ? `<ul class="tree">${data.taxonomy.map((node) => `
        <li style="margin-left:${(node.depth - 1) * 16}px">
          <span class="name">${esc(node.name)}</span>
          ${node.description ? `<div class="desc">${esc(node.description)}</div>` : ""}
        </li>`).join("")}</ul>`
    : `<p class="muted">The section hierarchy has not been turned into a taxonomy yet.
        "Map the taxonomy" asks the model to describe each branch — the branches themselves
        come from the LaTeX.</p>`;

  $("survey-note").innerHTML = data.note_html || '<p class="muted">No note yet.</p>';
  $("survey-analyze").textContent = data.has_note ? "Re-map the taxonomy" : "Map the taxonomy";
  $("survey-analyze").onclick = () =>
    runJob(api(`/papers/${encodeURIComponent(paperId)}/summarize`, { body: {} }),
           () => openSurvey(paperId));
  $("survey-resolve").onclick = () =>
    runJob(api(`/surveys/${encodeURIComponent(paperId)}/resolve`, { body: { limit: 30 } }),
           () => loadReferences());
  $("survey-refresh").onclick = () =>
    runJob(api(`/surveys/${encodeURIComponent(paperId)}/references/refresh`, { body: {} }),
           () => openSurvey(paperId));
  $("survey-harvest").onclick = () =>
    runJob(api(`/surveys/${encodeURIComponent(paperId)}/harvest`, {
      body: { limit: 10, section: $("survey-section").value || null },
    }), () => loadReferences());

  await loadReferences(true);
}

async function loadReferences(resetSections = false) {
  if (!currentSurvey) return;
  if (resetSections) {
    // Branch names differ per survey, so a filter left over from the last one
    // would silently return nothing.
    $("survey-section").innerHTML = '<option value="">Every branch</option>';
  }
  const section = $("survey-section").value;
  const limit = $("survey-limit").value || 40;
  const missing = $("survey-missing").checked;
  const data = await api(
    `/surveys/${encodeURIComponent(currentSurvey)}/references` +
    `?limit=${limit}&missing_only=${missing}&section=${encodeURIComponent(section)}`
  );

  if (resetSections) {
    const select = $("survey-section");
    select.innerHTML = '<option value="">Every branch</option>' +
      data.sections.map((name) => `<option value="${esc(name)}">${esc(name)}</option>`).join("");
  }

  $("survey-refs").innerHTML = data.references.length ? `
    <p class="small muted">${data.total} cited references · ${data.on_arxiv} have arXiv sources ·
      ${data.in_library} already in the library</p>
    <table>
      <thead><tr>
        <th class="num">Cites</th><th>Reference</th><th>Branch</th><th>arXiv</th><th></th>
      </tr></thead>
      <tbody>${data.references.map((reference) => `
        <tr>
          <td class="num">${reference.citations}${
            reference.table_citations ? `<span class="muted"> +${reference.table_citations}t</span>` : ""
          }</td>
          <td>${esc(reference.title || reference.key)}${
            reference.year ? ` <span class="muted">${esc(reference.year)}</span>` : ""
          }</td>
          <td class="muted small">${esc(reference.section)}</td>
          <td class="mono">${reference.arxiv_id
            ? `<a href="https://arxiv.org/abs/${esc(reference.arxiv_id)}" target="_blank" rel="noopener">${esc(reference.arxiv_id)}</a>`
            : '<span class="muted">—</span>'}</td>
          <td>${reference.in_library ? '<span class="tag">in library</span>' : ""}</td>
        </tr>`).join("")}</tbody>
    </table>` : '<div class="empty">No references match that filter.</div>';
}

/* ----------------------------------------------------------------- canon */

async function loadCanon() {
  const min = $("canon-min").value || 2;
  const data = await api(`/core?min_surveys=${min}`);
  const table = $("canon-table");
  if (!data.references.length) {
    table.innerHTML = `<div class="empty">Nothing is cited by ${min} or more surveys yet.
      Import a second survey of the same field and this fills in.</div>`;
    return;
  }
  table.innerHTML = `
    <p class="small muted">Across ${data.surveys.length} survey(s) in the library.</p>
    <table>
      <thead><tr>
        <th class="num">Surveys</th><th class="num">Cites</th><th>Reference</th>
        <th>arXiv</th><th></th>
      </tr></thead>
      <tbody>${data.references.map((reference) => `
        <tr>
          <td class="num"><strong>${reference.agreeing}</strong></td>
          <td class="num">${reference.citations}</td>
          <td>${esc(reference.title || reference.key)}</td>
          <td class="mono"><a href="https://arxiv.org/abs/${esc(reference.arxiv_id)}"
            target="_blank" rel="noopener">${esc(reference.arxiv_id)}</a></td>
          <td>${reference.in_library
            ? '<span class="tag">in library</span>'
            : `<button class="btn link" data-import="${esc(reference.arxiv_id)}">import</button>`}</td>
        </tr>`).join("")}</tbody>
    </table>`;
  table.querySelectorAll("[data-import]").forEach((node) => {
    node.onclick = () => runJob(
      api("/ingest", { body: { text: node.dataset.import, kind: "paper" } }),
      loadCanon
    );
  });
}

/* ----------------------------------------------------------- collections */

/* Chosen so it cannot collide with a collection someone actually names. */
const UNFILED = "\u0000unfiled";

let filePicker;

async function loadCollections() {
  const data = await api("/collections");
  state.collections = data.collections;

  const options = (extra) => '<option value="">' + extra + "</option>" +
    data.collections.map((item) =>
      `<option value="${esc(item.name)}">${esc(item.name)} (${item.papers})</option>`).join("");

  const filter = $("library-collection");
  const kept = filter.value;
  filter.innerHTML = options("All collections") +
    (data.unfiled ? `<option value="${UNFILED}">Unfiled (${data.unfiled})</option>` : "");
  filter.value = kept;
  if (filter.value !== kept) filter.value = "";

  const scope = $("knowledge-collection");
  const keptScope = scope.value;
  scope.innerHTML = options("The whole library");
  scope.value = keptScope;
  if (scope.value !== keptScope) scope.value = "";

  $("collection-names").innerHTML = data.collections
    .map((item) => `<option value="${esc(item.name)}">`).join("");
}

async function filePapers(name) {
  const papers = filePicker.selected();
  if (!papers.length) return toast("Select the papers to file first.", true);
  const data = await api("/collections/assign", { body: { papers, collection: name } });
  toast(name
    ? `Filed ${data.moved.length} paper(s) under ${name}`
    : `Unfiled ${data.moved.length} paper(s)`);
  $("file-panel").classList.add("hidden");
  filePicker.setSelection([]);
  await loadLibrary();
}

/* ---------------------------------------------------------- paper picker */

/* A searchable, multi-select list of papers. Used for the scope of a question
   and for putting papers into a collection, so it takes its rows from whatever
   list it is handed rather than fetching its own. */
function createPicker(rootId, { onChange, emptyLabel = "the whole library" } = {}) {
  const root = $(rootId);
  const search = root.querySelector(".picker-search");
  const list = root.querySelector(".picker-list");
  const chips = root.querySelector(".picker-chips");
  const count = root.querySelector(".picker-count");
  const clear = root.querySelector(".picker-clear");

  let papers = [];
  const chosen = new Map();

  const matches = (paper, needle) =>
    !needle ||
    `${paper.title} ${paper.paper_id} ${paper.authors || ""}`.toLowerCase().includes(needle);

  function draw() {
    const needle = search.value.trim().toLowerCase();
    const shown = papers.filter((paper) => matches(paper, needle));
    list.innerHTML = shown.length
      ? shown.slice(0, 200).map((paper) => `
          <label class="picker-row${chosen.has(paper.paper_id) ? " on" : ""}">
            <input type="checkbox" value="${esc(paper.paper_id)}"
                   ${chosen.has(paper.paper_id) ? "checked" : ""}>
            <span class="picker-title">${esc(paper.title)}</span>
            <span class="picker-meta mono">${esc(paper.paper_id)}</span>
            ${paper.kind === "survey" ? '<span class="tag">survey</span>' : ""}
          </label>`).join("")
      : '<div class="empty small">Nothing matches that.</div>';

    list.querySelectorAll("input[type=checkbox]").forEach((box) => {
      box.onchange = () => {
        const paper = papers.find((item) => item.paper_id === box.value);
        if (box.checked && paper) chosen.set(paper.paper_id, paper);
        else chosen.delete(box.value);
        draw();
        onChange?.(selected());
      };
    });

    chips.innerHTML = [...chosen.values()].map((paper) => `
      <span class="chip" data-drop="${esc(paper.paper_id)}">
        ${esc(paper.title.slice(0, 46))}${paper.title.length > 46 ? "…" : ""}
        <span class="x">×</span>
      </span>`).join("");
    chips.querySelectorAll("[data-drop]").forEach((chip) => {
      chip.onclick = () => {
        chosen.delete(chip.dataset.drop);
        draw();
        onChange?.(selected());
      };
    });

    count.textContent = chosen.size
      ? `${chosen.size} selected`
      : `none selected — ${emptyLabel}`;
    clear.classList.toggle("hidden", chosen.size === 0);
  }

  const selected = () => [...chosen.keys()];

  search.oninput = draw;
  clear.onclick = () => {
    chosen.clear();
    draw();
    onChange?.(selected());
  };

  return {
    selected,
    setPapers(next) {
      papers = next || [];
      const live = new Set(papers.map((paper) => paper.paper_id));
      [...chosen.keys()].filter((id) => !live.has(id)).forEach((id) => chosen.delete(id));
      draw();
    },
    setSelection(ids) {
      chosen.clear();
      (ids || []).forEach((id) => {
        const paper = papers.find((item) => item.paper_id === id);
        if (paper) chosen.set(id, paper);
      });
      draw();
    },
  };
}

/* ------------------------------------------------------------------- ask */

let askPicker;

function fillScopeSelect(papers) {
  askPicker?.setPapers(papers);
}

function answerCard(title, html, sources) {
  const card = document.createElement("div");
  card.className = "card prose";
  card.innerHTML =
    `<h3>${esc(title)}</h3>${html}` +
    (sources?.length
      ? `<p class="small muted">Sources: ${sources.map(esc).join(" · ")}</p>`
      : "");
  $("ask-answers").prepend(card);
}

function askQuestion() {
  const question = $("ask-question").value.trim();
  if (!question) return toast("Ask something first.", true);
  const papers = askPicker.selected();
  runJob(api("/ask", { body: { question, papers: papers.length ? papers : null } }), (job) => {
    if (!job.result) return;
    answerCard(job.result.question, job.result.html, job.result.sources);
    $("ask-question").value = "";
  });
}

function comparePapers() {
  const papers = askPicker.selected();
  if (papers.length < 2) return toast("Select at least two papers to compare.", true);
  const aspect = $("ask-question").value.trim();
  runJob(api("/compare", { body: { papers, aspect } }), (job) => {
    if (!job.result) return;
    answerCard(aspect || `Comparing ${papers.length} papers`, job.result.html, papers);
  });
}

/* ------------------------------------------------------------- knowledge */

async function loadKnowledge() {
  const collection = $("knowledge-collection").value;
  await loadCollections();
  $("knowledge-collection").value = collection;
  const data = await api(
    "/knowledge" + (collection ? `?collection=${encodeURIComponent(collection)}` : "")
  );
  const list = $("knowledge-list");
  if (!data.documents.length) {
    list.innerHTML = `<div class="empty">Nothing written yet. The buttons above build these
      from the notes you already have.</div>`;
    return;
  }

  // Group by the folder each document sits in, so topics/ and fields/ read as
  // sections rather than as a flat list of paths.
  const groups = new Map();
  data.documents.forEach((document_) => {
    const folder = document_.folder || "";
    if (!groups.has(folder)) groups.set(folder, []);
    groups.get(folder).push(document_);
  });

  list.innerHTML = [...groups.entries()].map(([folder, documents]) => `
    ${folder ? `<h3 class="group">${esc(folder)}</h3>` : ""}
    ${documents.map((document_) => `
      <button class="item" data-doc="${esc(document_.name)}">
        <div class="title">${esc(document_.title)}</div>
        <div class="meta mono">${esc(document_.name)}</div>
      </button>`).join("")}`).join("");

  list.querySelectorAll("[data-doc]").forEach((node) => {
    node.onclick = () => openDoc(node.dataset.doc);
  });
}

async function openDoc(name, remember = true) {
  let doc;
  try {
    doc = await api(`/knowledge/doc?name=${encodeURIComponent(name)}`);
  } catch (error) {
    toast(error.message, true);
    return show("knowledge");
  }
  $("doc-body").innerHTML = doc.html;
  show("doc", false);
  if (remember) setHash("doc/" + encodeURIComponent(name));
}

/* -------------------------------------------------------------- settings */

const PRESETS = [
  { name: "DeepSeek", url: "https://api.deepseek.com", model: "deepseek-v4-flash" },
  { name: "OpenAI", url: "https://api.openai.com/v1", model: "gpt-5.6-terra" },
  { name: "通义千问 Qwen (DashScope)", url: "https://dashscope.aliyuncs.com/compatible-mode/v1", model: "qwen-plus" },
  { name: "Moonshot Kimi", url: "https://api.moonshot.cn/v1", model: "kimi-k3" },
  { name: "智谱 GLM", url: "https://open.bigmodel.cn/api/paas/v4", model: "glm-5.2" },
  { name: "OpenRouter", url: "https://openrouter.ai/api/v1", model: "~openai/gpt-latest" },
  { name: "Ollama (local)", url: "http://localhost:11434/v1", model: "gpt-oss:20b" },
  { name: "vLLM (local)", url: "http://localhost:8000/v1", model: "" },
];

function fillSettings() {
  const settings = state.settings;
  if (!settings) return;
  $("set-root").value = settings.root;
  $("set-language").value = settings.default_language;
  $("set-base-url").value = settings.llm.base_url;
  $("set-model").value = settings.llm.model;
  $("set-fast-model").value = settings.llm.fast_model;
  $("set-temperature").value = settings.llm.temperature;
  $("set-timeout").value = settings.llm.timeout;
  $("set-context").value = settings.llm.max_context_chars;
  $("set-user-agent").value = settings.arxiv.user_agent;
  $("set-interval").value = settings.arxiv.request_interval;
  $("set-api-key").placeholder = settings.llm.has_key
    ? "a key is saved — leave blank to keep it"
    : `stored in ${settings.llm.api_key_env}`;
  $("settings-paths").textContent = `${settings.root}/config.toml and ${settings.root}/.env`;

  const select = $("set-preset");
  select.innerHTML = '<option value="">Custom</option>' +
    PRESETS.map((preset, index) => `<option value="${index}">${esc(preset.name)}</option>`).join("");
  const match = PRESETS.findIndex((preset) => preset.url === settings.llm.base_url);
  select.value = match >= 0 ? String(match) : "";

  const bots = settings.bots || {};
  $("set-feishu-enabled").checked = !!bots.feishu_enabled;
  $("set-wecom-enabled").checked = !!bots.wecom_enabled;
  $("set-feishu-state").textContent = bots.feishu_app_id && bots.has_feishu_secret
    ? "— app credentials found"
    : "— needs FEISHU_APP_ID and FEISHU_APP_SECRET in .env";
  $("set-wecom-state").textContent = bots.has_wecom_bot
    ? "— smart robot credentials found"
    : bots.has_wecom_token
      ? "— self-built app credentials found"
      : "— needs WECOM_BOT_ID, or WECOM_TOKEN and the rest, in .env";
  $("set-send-as-image").value = bots.send_as_image || "auto";
  $("set-image-state").textContent = bots.can_draw
    ? "Neither platform draws Markdown tables or formulas, so a reply that has them is rendered to an image first."
    : "No Chrome-family browser was found, so replies are sent as text whatever this says. Set SURVEYOR_BROWSER to point at one.";
  fillBotsHint();
}

/* The switch records the intent; starting the bot is still a terminal command,
   because it outlives the page. Say which one, so nobody has to guess. */
function fillBotsHint() {
  const bots = state.settings?.bots || {};
  const lines = [];
  if ($("set-feishu-enabled").checked) {
    lines.push("Feishu: run `surveyor feishu-connect` — a long connection, so no public URL is needed.");
  }
  if ($("set-wecom-enabled").checked) {
    lines.push(bots.has_wecom_bot
      ? "WeCom: run `surveyor wecom-connect` — a long connection, so no public URL is needed."
      : `WeCom: run \`surveyor serve\` on port ${bots.port} behind a public HTTPS URL.`);
  }
  $("set-bots-hint").textContent = lines.join("  ") ||
    "Both are off, so no bot answers messages.";
}

async function saveSettings() {
  const key = $("set-api-key").value.trim();
  const payload = {
    root: $("set-root").value.trim(),
    default_language: $("set-language").value,
    llm: {
      base_url: $("set-base-url").value.trim(),
      model: $("set-model").value.trim(),
      fast_model: $("set-fast-model").value.trim() || null,
      temperature: Number($("set-temperature").value),
      timeout: Number($("set-timeout").value),
      max_context_chars: Number($("set-context").value),
      api_key_env: state.settings.llm.api_key_env,
    },
    arxiv: {
      user_agent: $("set-user-agent").value.trim(),
      request_interval: Number($("set-interval").value),
    },
    feishu: {
      enabled: $("set-feishu-enabled").checked,
      send_as_image: $("set-send-as-image").value,
    },
    wecom: {
      enabled: $("set-wecom-enabled").checked,
      send_as_image: $("set-send-as-image").value,
    },
    secrets: key ? { [state.settings.llm.api_key_env]: key } : {},
  };
  try {
    const data = await api("/settings", { body: payload });
    state.settings = data.settings;
    $("set-api-key").value = "";
    $("set-saved").textContent = "Saved.";
    setTimeout(() => ($("set-saved").textContent = ""), 3000);
    await refreshState();
    fillSettings();
    $("setup-banner").classList.toggle("hidden", data.settings.llm.has_key);
  } catch (error) {
    toast(error.message, true);
  }
}

async function testConnection() {
  const box = $("set-test-result");
  box.className = "result";
  box.textContent = "Calling the endpoint…";
  const result = await api("/settings/test", {
    body: { base_url: $("set-base-url").value.trim(), model: $("set-model").value.trim() },
  });
  box.className = "result " + (result.ok ? "ok" : "bad");
  box.textContent = result.message;
}

/* ------------------------------------------------------------------ wire */

function wire() {
  document.querySelectorAll("#nav button").forEach((button) => {
    button.onclick = () => show(button.dataset.view);
  });
  document.querySelectorAll("[data-back]").forEach((button) => {
    button.onclick = () => show(button.dataset.back);
  });

  $("import-go").onclick = () => {
    const text = $("import-text").value.trim();
    if (!text) return toast("Paste at least one arXiv id or link.", true);
    runJob(api("/ingest", {
      body: {
        text,
        kind: $("import-kind").value || null,
        collection: $("import-collection").value.trim(),
        summarize: $("import-summarize").checked,
      },
    }), () => {
      $("import-text").value = "";
      loadLibrary();
      refreshState();
    });
  };

  let filterTimer;
  const refilter = () => {
    clearTimeout(filterTimer);
    filterTimer = setTimeout(loadLibrary, 200);
  };
  $("library-search").oninput = refilter;
  $("library-kind").onchange = loadLibrary;

  $("survey-section").onchange = () => loadReferences();
  $("survey-limit").onchange = () => loadReferences();
  $("survey-missing").onchange = () => loadReferences();

  $("canon-refresh").onclick = loadCanon;
  $("canon-min").onchange = loadCanon;

  askPicker = createPicker("ask-picker");
  $("ask-go").onclick = askQuestion;
  $("ask-compare").onclick = comparePapers;
  $("ask-question").onkeydown = (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) askQuestion();
  };

  filePicker = createPicker("file-picker", { emptyLabel: "nothing to file" });
  $("library-collection").onchange = loadLibrary;
  $("library-file").onclick = () => {
    const panel = $("file-panel");
    panel.classList.toggle("hidden");
    if (!panel.classList.contains("hidden")) $("file-name").focus();
  };
  $("file-go").onclick = () => filePapers($("file-name").value.trim());
  $("file-clear").onclick = () => filePapers("");

  $("knowledge-collection").onchange = loadKnowledge;
  document.querySelectorAll("[data-build]").forEach((button) => {
    button.onclick = () => {
      const what = button.dataset.build;
      const collection = $("knowledge-collection").value;
      const request = what === "merge"
        ? api("/surveys/merge", { body: { collection } })
        : api("/knowledge/build", { body: { what, collection } });
      runJob(request, loadKnowledge);
    };
  });

  $("set-preset").onchange = () => {
    const preset = PRESETS[Number($("set-preset").value)];
    if (!preset) return;
    $("set-base-url").value = preset.url;
    if (preset.model) $("set-model").value = preset.model;
  };
  $("set-feishu-enabled").onchange = fillBotsHint;
  $("set-wecom-enabled").onchange = fillBotsHint;
  $("set-save").onclick = saveSettings;
  $("set-test").onclick = () => testConnection().catch((error) => toast(error.message, true));
}

async function main() {
  wire();
  const data = await refreshState();
  if (!data.configured) {
    $("setup-banner").classList.remove("hidden");
    show("settings");
    return;
  }
  route(location.hash || "#library");
}

main().catch((error) => toast(error.message, true));
