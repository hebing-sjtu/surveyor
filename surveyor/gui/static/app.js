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

const state = { settings: null, stats: null, papers: [], view: "library" };

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
  const data = await api(`/papers?kind=${kind}&q=${encodeURIComponent(query)}`);
  state.papers = data.papers;
  fillScopeSelect(data.papers);

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
  $("paper-note").innerHTML = data.note_html ||
    `<p class="muted">No note yet. ${esc(data.abstract)}</p>`;

  const link = $("paper-arxiv");
  link.href = paper.abs_url || `https://arxiv.org/abs/${paper.arxiv_id}`;
  link.classList.toggle("hidden", !paper.arxiv_id && !paper.abs_url);

  $("paper-summarize").textContent = paper.has_note ? "Rewrite the note" : "Write the note";
  $("paper-summarize").onclick = () =>
    runJob(api(`/papers/${encodeURIComponent(paperId)}/summarize`, { body: {} }),
           () => openPaper(paperId));
  $("paper-ask").onclick = () => {
    show("ask");
    $("ask-scope").value = paperId;
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

/* ------------------------------------------------------------------- ask */

function fillScopeSelect(papers) {
  const select = $("ask-scope");
  const chosen = select.value;
  select.innerHTML = '<option value="">The whole library</option>' +
    papers.map((paper) =>
      `<option value="${esc(paper.paper_id)}">${esc(paper.title.slice(0, 70))}</option>`).join("");
  select.value = chosen;
}

function askQuestion() {
  const question = $("ask-question").value.trim();
  if (!question) return;
  const scope = $("ask-scope").value;
  runJob(api("/ask", { body: { question, papers: scope ? [scope] : null } }), (job) => {
    const answer = job.result;
    if (!answer) return;
    const card = document.createElement("div");
    card.className = "card prose";
    card.innerHTML =
      `<h3>${esc(answer.question)}</h3>${answer.html}` +
      (answer.sources?.length
        ? `<p class="small muted">Sources: ${answer.sources.map(esc).join(" · ")}</p>`
        : "");
    $("ask-answers").prepend(card);
    $("ask-question").value = "";
  });
}

/* ------------------------------------------------------------- knowledge */

async function loadKnowledge() {
  const data = await api("/knowledge");
  const list = $("knowledge-list");
  if (!data.documents.length) {
    list.innerHTML = `<div class="empty">Nothing written yet. The buttons above build these
      from the notes you already have.</div>`;
    return;
  }
  list.innerHTML = data.documents.map((document_) => `
    <button class="item" data-doc="${esc(document_.name)}">
      <div class="title">${esc(document_.title)}</div>
      <div class="meta mono">${esc(document_.name)}</div>
    </button>`).join("");
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
    feishu: { enabled: $("set-feishu-enabled").checked },
    wecom: { enabled: $("set-wecom-enabled").checked },
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
        summarize: $("import-summarize").checked,
      },
    }), () => {
      $("import-text").value = "";
      loadLibrary();
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

  $("ask-go").onclick = askQuestion;
  $("ask-question").onkeydown = (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) askQuestion();
  };

  document.querySelectorAll("[data-build]").forEach((button) => {
    button.onclick = () => {
      const what = button.dataset.build;
      const request = what === "merge"
        ? api("/surveys/merge", { body: {} })
        : api("/knowledge/build", { body: { what } });
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
