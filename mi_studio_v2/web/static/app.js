"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  mode: "train",
  health: null,
  sources: [],
  outputs: [],
  outputsConfirmed: false,
  model: null,
  profileLocked: false,
  job: null,
  logOffset: 0,
  pollTimer: null,
  lastResult: null,
  browser: { mode: "data", path: "", parent: null, entries: [] },
  busy: false,
};

const KIND_CLASS = {
  cargame_record: "cargame",
  offline_session: "offline",
  online_session: "online",
  online_calibration_session: "online_calibration_session",
};
const KIND_LABEL = {
  cargame_record: "cargame 记录",
  offline_session: "离线训练",
  online_session: "在线数据",
  online_calibration_session: "在线校准",
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function fmtDur(seconds) {
  if (seconds == null) return "—";
  const s = Number(seconds);
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m${Math.round(s - m * 60)}s`;
}

async function api(path, options = {}) {
  const opts = { headers: { "Content-Type": "application/json" }, ...options };
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { error: text }; }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

function toast(message, kind = "info") {
  const note = $("#run-note");
  note.textContent = message || "";
  note.className = kind === "bad" ? "bad" : kind === "warn" ? "warn" : "muted";
  if (message) {
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { note.textContent = ""; note.className = "muted"; }, 7000);
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  try {
    const health = await api("/api/health");
    state.health = health;
    const backend = $("#backend");
    backend.textContent = `在线 · ${health.version}`;
    backend.className = "pill ok";
    state.outputs = (health.default_outputs || []).map((o) => ({ ...o }));
    const profileSel = $("#profile");
    profileSel.innerHTML = Object.values(health.profiles || {}).map((p) =>
      `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join("");
    profileSel.value = health.default_profile;
    renderOutputs();
    renderSources();
    renderJobs(health.jobs || []);
    updatePreview();
    if(health.default_source_model){setMode('calibrate');await loadModel(health.default_source_model);}
    setInterval(refreshJobs, 20000);
  } catch (error) {
    const backend = $("#backend");
    backend.textContent = "后端不可用";
    backend.className = "pill bad";
    toast(error.message, "bad");
  }
}

// ---------------------------------------------------------------------------
// mode / outputs
// ---------------------------------------------------------------------------
function setMode(mode) {
  state.mode = mode;
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  const calibrating = mode === "calibrate";
  $("#model-card").hidden = !calibrating;
  $("#cal-params").hidden = !calibrating;
  $("#train-params").hidden = calibrating;
  $("#run-mode-badge").textContent = calibrating ? "源校准" : "源生成";
  $("#mode-hint").textContent = calibrating
    ? "读入一个六类源模型 + 校准数据。6 维输出语义从源模型继承并锁定；只重拟合最后的线性头。"
    : "从多份 Session 训练一个六类源模型。必须逐 Session 设置标签映射，并显式声明 6 个输出维度的含义。";
  if (!calibrating) {
    state.model = null;
    state.profileLocked = false;
    $("#legacy-profile-wrap").hidden = true;
    $("#model-info").innerHTML = "";
    $("#outputs-lock").textContent = "必须显式确认";
    $("#outputs-lock").className = "badge lock";
    if (!state.outputsConfirmed) $("#outputs-confirm").checked = false;
  } else {
    state.outputsConfirmed = false;
    $("#outputs-confirm").checked = false;
    $("#outputs-lock").textContent = state.model ? "已锁定（继承源模型）" : "载入源模型后锁定";
    $("#outputs-lock").className = "badge lock";
  }
  renderOutputs();
  renderSources();
  updatePreview();
}

function renderOutputs() {
  const calibrating = state.mode === "calibrate";
  const locked = calibrating && state.model != null;
  const box = $("#outputs");
  box.innerHTML = state.outputs.map((out, index) => `
    <div class="out-row ${locked ? "out-locked" : ""}">
      <div class="out-dim">${index}</div>
      <input data-out="${index}" data-field="name" value="${esc(out.name)}"
             placeholder="含义（如 左握拳）" ${locked ? "disabled" : ""}>
      <input data-out="${index}" data-field="label_id" type="number" value="${esc(out.label_id)}"
             title="模型标签 ID" ${locked ? "disabled" : ""}>
      <input data-out="${index}" data-field="description" value="${esc(out.description || "")}"
             placeholder="备注（可选）" ${locked ? "disabled" : ""}>
    </div>`).join("") + `<p class="hint">列顺序 = 模型输出顺序（model_labels = [${state.outputs.map((o) => o.label_id).join(", ")}]）。</p>`;
  box.querySelectorAll("input").forEach((input) => {
    input.addEventListener("input", () => {
      const i = Number(input.dataset.out);
      const field = input.dataset.field;
      state.outputs[i][field] = field === "label_id" ? Number(input.value) : input.value;
      state.outputsConfirmed = false;
      $("#outputs-confirm").checked = false;
      updatePreview();
    });
  });
  if (locked) {
    state.outputsConfirmed = true;
    $("#outputs-confirm").checked = true;
    $("#outputs-confirm").disabled = true;
    $("#outputs-lock").textContent = `已锁定 · ${state.model.payload?.studio?.output_fingerprint || "继承源模型"}`;
  } else {
    $("#outputs-confirm").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------
function suggestTarget(hint) {
  if (!hint) return null;
  const clean = String(hint).replace(/[（(].*$/, "").trim();
  const exact = state.outputs.findIndex((o) => o.name === clean || o.name === hint);
  if (exact >= 0) return exact;
  const contains = state.outputs.findIndex((o) =>
    (o.name && (clean.includes(o.name) || o.name.includes(clean))));
  return contains >= 0 ? contains : null;
}

function rowsFromScan(row) {
  return (row.labels || []).map((label) => {
    const target = suggestTarget(label.hint);
    return {
      id: label.id,
      hint: label.hint || null,
      hint_source: label.hint_source || null,
      declared: label.declared || null,
      count: label.count || 0,
      target,
      unused: false,
      reviewed: false,
    };
  });
}

async function scanPath(path) {
  if (!path) { toast("请输入目录路径", "warn"); return; }
  try {
    const data = await api("/api/scan", { method: "POST", body: JSON.stringify({ path }) });
    const rows = (data.sessions || []).filter((row) => row.kind !== "invalid");
    const invalid = (data.sessions || []).filter((row) => row.kind === "invalid");
    if (!rows.length) {
      toast(invalid[0]?.error || "没有识别到会话", "bad");
      return;
    }
    let added = 0;
    for (const row of rows) {
      if (state.sources.some((s) => s.path === row.path)) continue;
      state.sources.push({
        path: row.path,
        scan: row,
        rows: rowsFromScan(row),
        expanded: true,
      });
      added += 1;
    }
    renderSources();
    updatePreview();
    toast(added ? `已加入 ${added} 个 Session` : "这些 Session 已经在列表里", added ? "info" : "warn");
  } catch (error) {
    toast(error.message, "bad");
  }
}

function reviewAll(source) {
  source.rows.forEach((row) => {
    if (row.target != null || row.unused) row.reviewed = true;
  });
  renderSources();
  updatePreview();
}

function confirmAllSources() {
  state.sources.forEach((source) => source.rows.forEach((row) => {
    if (row.target != null || row.unused) row.reviewed = true;
  }));
  renderSources();
  updatePreview();
}

function applyMappingToAll(source) {
  const template = new Map(source.rows.map((row) => [row.id, row]));
  state.sources.forEach((other) => {
    if (other === source) return;
    other.rows.forEach((row) => {
      const match = template.get(row.id);
      if (!match) return;
      row.target = match.target;
      row.unused = match.unused;
      row.reviewed = match.reviewed;
    });
  });
  renderSources();
  updatePreview();
  toast("映射已按源标签 ID 复制到其它 Session", "info");
}

function renderSources() {
  const list = $("#source-list");
  if (!state.sources.length) {
    list.innerHTML = `<div class="muted" style="padding:8px 2px">还没有数据。扫描一个 Session 目录开始。</div>`;
    $("#src-count").textContent = "0 个 Session";
    return;
  }
  const total = state.sources.length;
  const reviewed = state.sources.reduce((n, s) => n + s.rows.filter((r) => r.reviewed).length, 0);
  const all = state.sources.reduce((n, s) => n + s.rows.length, 0);
  $("#src-count").textContent = `${total} 个 Session · 已确认 ${reviewed}/${all}`;
  list.innerHTML = state.sources.map((source, si) => {
    const row = source.scan;
    const kindClass = KIND_CLASS[row.kind] || "";
    const labelRows = source.rows.map((item, ri) => {
      const selected = item.unused ? "unused" : item.target == null ? "" : String(item.target);
      const options = [
        `<option value="" ${selected === "" ? "selected" : ""}>— 未选择 —</option>`,
        `<option value="unused" ${selected === "unused" ? "selected" : ""}>不使用</option>`,
        ...state.outputs.map((out, oi) =>
          `<option value="${oi}" ${selected === String(oi) ? "selected" : ""}>${oi} · ${esc(out.name)}${out.label_id != null ? ` (id ${esc(out.label_id)})` : ""}</option>`),
      ].join("");
      const hint = item.hint ? esc(item.hint) : `<span class="muted">无元数据提示</span>`;
      return `<tr class="${item.reviewed ? "" : "row-unreviewed"} ${item.unused ? "row-unused" : ""}">
        <td><b>${esc(item.id)}</b></td>
        <td>${hint}${item.hint_source ? `<span class="suggest">${esc(item.hint_source)}</span>` : ""}${item.declared && item.declared !== item.hint ? `<div class="muted">类别表：${esc(item.declared)}</div>` : ""}</td>
        <td class="count">${esc(item.count)}</td>
        <td><select data-src="${si}" data-row="${ri}">${options}</select></td>
        <td><input class="reviewed" type="checkbox" data-review="${si}:${ri}" ${item.reviewed ? "checked" : ""} title="确认这一行"></td>
        <td>${item.reviewed ? `<span class="ok">已确认</span>` : `<span class="warn">待复核</span>`}</td>
      </tr>`;
    }).join("");
    const warns = (row.warnings || []).map((w) => `<div class="warn" style="font-size:11.5px">⚠ ${esc(w)}</div>`).join("");
    return `<div class="source">
      <div class="source-head">
        <span class="kind ${kindClass}">${esc(row.kind_label || "")}</span>
        <span class="source-title">${esc(row.session_id)}</span>
        <span class="source-sub">${esc(row.date || "?")} · ${row.n_trials} 试次 · ${fmtDur(row.duration_s)} · ${row.n_channels} ch</span>
        <div class="source-actions">
          <button class="ghost small" data-act="review" data-src="${si}">全部确认</button>
          <button class="ghost small" data-act="apply" data-src="${si}">映射到全部</button>
          <button class="ghost small" data-act="remove" data-src="${si}">移除</button>
        </div>
      </div>
      ${warns}
      <table class="label-table">
        <thead><tr><th>源标签 id</th><th>元数据提示</th><th>试次</th><th>映射到输出列</th><th>复核</th><th>状态</th></tr></thead>
        <tbody>${labelRows}</tbody>
      </table>
    </div>`;
  }).join("");

  list.querySelectorAll("select[data-src]").forEach((sel) => {
    sel.addEventListener("change", () => {
      const si = Number(sel.dataset.src);
      const ri = Number(sel.dataset.row);
      const item = state.sources[si].rows[ri];
      if (sel.value === "unused") {
        item.target = null; item.unused = true;
      } else if (sel.value === "") {
        item.target = null; item.unused = false;
      } else {
        item.target = Number(sel.value); item.unused = false;
      }
      item.reviewed = false;
      renderSources(); updatePreview();
    });
  });
  list.querySelectorAll("input[data-review]").forEach((box) => {
    box.addEventListener("change", () => {
      const [si, ri] = box.dataset.review.split(":").map(Number);
      const item = state.sources[si].rows[ri];
      if (box.checked && item.target == null && !item.unused) {
        box.checked = false;
        toast("先选择目标列或「不使用」，再确认这一行", "warn");
        return;
      }
      item.reviewed = box.checked;
      renderSources(); updatePreview();
    });
  });
  list.querySelectorAll("button[data-act]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const si = Number(btn.dataset.src);
      if (btn.dataset.act === "review") reviewAll(state.sources[si]);
      if (btn.dataset.act === "apply") applyMappingToAll(state.sources[si]);
      if (btn.dataset.act === "remove") {
        state.sources.splice(si, 1);
        renderSources(); updatePreview();
      }
    });
  });
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------
async function loadModel(path) {
  if (!path) { toast("请输入模型路径", "warn"); return; }
  try {
    const info = await api(`/api/model?path=${encodeURIComponent(path)}`);
    info.path = info.path || info.model;
    state.model = info;
    state.outputs = info.outputs.map((o) => ({ ...o }));
    state.outputsConfirmed = true;
    $("#model-input").value = info.path;
    const studio = info.studio || {};
    const profileId = info.profile?.resolved || studio.profile?.id;
    if (profileId) {
      $("#profile").value = profileId;
      state.profileLocked = true;
      $("#profile").disabled = true;
      $("#legacy-profile-wrap").hidden = true;
    } else {
      state.profileLocked = false;
      $("#profile").disabled = false;
      $("#legacy-profile-wrap").hidden = false;
    }
    $("#model-info").innerHTML = `
      <div><span class="ok">已载入</span> · ${esc(info.path)}</div>
      <div style="margin-top:6px">
        ${info.outputs.map((o) => `<span class="chip">${o.index}: ${esc(o.name)} (id ${esc(o.label_id)})</span>`).join("")}
      </div>
      <div class="muted" style="margin-top:6px">arch: ${esc(JSON.stringify(info.arch))} · ${info.calibrated ? "已校准" : "未校准"}</div>
      ${studio.output_fingerprint ? `<div class="muted">输出指纹：${esc(studio.output_fingerprint)}</div>` : ""}`;
    // re-suggest unmapped rows against the new output contract
    state.sources.forEach((source) => source.rows.forEach((row) => {
      if (!row.reviewed) row.target = suggestTarget(row.hint);
    }));
    renderOutputs();
    renderSources();
    updatePreview();
    toast("源模型已载入，6 维输出已锁定", "info");
  } catch (error) {
    toast(error.message, "bad");
  }
}

// ---------------------------------------------------------------------------
// config / preview
// ---------------------------------------------------------------------------
function buildConfig({ strict = true } = {}) {
  if (!state.sources.length) throw new Error("至少加入一个 Session");
  const unreviewed = [];
  state.sources.forEach((source) => source.rows.forEach((row) => {
    if (!row.reviewed) unreviewed.push(`${source.scan.session_id} / 标签 ${row.id}`);
  }));
  if (strict && unreviewed.length) {
    throw new Error(`还有 ${unreviewed.length} 行未复核：${unreviewed.slice(0, 3).join("；")}…`);
  }
  if (strict && !state.outputsConfirmed) throw new Error("请先确认 6 维输出语义");
  const names = state.outputs.map((o) => String(o.name || "").trim());
  if (names.some((n) => !n)) throw new Error("6 维输出的名称不能为空");
  const ids = state.outputs.map((o) => Number(o.label_id));
  if (new Set(ids).size !== 6) throw new Error("6 维输出的 label_id 必须互不相同");

  const usedTargets = new Map();
  state.sources.forEach((source) => source.rows.forEach((row) => {
    if (row.unused || row.target == null) return;
    if (!usedTargets.has(row.target)) usedTargets.set(row.target, []);
    usedTargets.get(row.target).push(row);
  }));
  const many = [...usedTargets.entries()].filter(([, rows]) => rows.length > 1);
  const manyConfirmed = $("#many-confirm").checked;
  if (strict && many.length && !manyConfirmed) {
    throw new Error("存在多对一映射；请勾选运行区的确认框");
  }

  const config = {
    kind: state.mode,
    profile: $("#profile").value,
    name: $("#job-name").value.trim() || (state.mode === "train" ? "六类源模型" : "六类校准模型"),
    sources: state.sources.map((source) => ({
      path: source.path,
      mapping: {
        rows: source.rows.map((row) => ({
          id: row.id,
          target: row.unused ? null : row.target,
          unused: Boolean(row.unused),
          reviewed: Boolean(row.reviewed),
          hint: row.hint || null,
        })),
        many_to_one_confirmed: many.length ? manyConfirmed : false,
      },
    })),
    outputs: state.outputs.map((o, i) => ({
      index: i, label_id: Number(o.label_id), name: String(o.name).trim(),
      description: String(o.description || "").trim(),
    })),
  };
  if (state.mode === "train") {
    config.train = {
      pre_epochs: Number($("#pre-epochs").value),
      head_epochs: Number($("#head-epochs").value),
      ft_epochs: Number($("#ft-epochs").value),
      patience: Number($("#patience").value),
      batch: Number($("#batch").value),
      hidden: 8,
      seed: Number($("#seed").value),
      device: $("#device").value,
      eval: $("#eval-mode").value,
      final_fit: $("#final-fit").checked,
    };
    config.export_onnx = $("#export-onnx-train").checked;
  } else {
    if (!state.model) throw new Error("请先载入源模型");
    config.source_model = state.model.path;
    config.calibration = {
      C: Number($("#cal-c").value),
      holdout_per_class: Number($("#cal-holdout").value),
      seed: Number($("#seed").value),
      device: $("#device").value,
    };
    config.legacy_profile_confirmed = Boolean($("#legacy-profile-confirm")?.checked);
    config.export_onnx = $("#export-onnx-cal").checked;
  }
  return { config, many };
}

function updatePreview() {
  try {
    const { config } = buildConfig({ strict: false });
    const text = JSON.stringify(config, null, 2);
    $("#cli-preview").textContent =
      `# 保存为 job.json，然后：\n$ python -m mi_studio run --config job.json\n\n${text}`;
    $("#many-wrap").hidden = !hasManyToOne();
  } catch (error) {
    $("#cli-preview").textContent = `# ${error.message}`;
    $("#many-wrap").hidden = true;
  }
}

function hasManyToOne() {
  const seen = new Map();
  state.sources.forEach((source) => source.rows.forEach((row) => {
    if (row.unused || row.target == null) return;
    seen.set(row.target, (seen.get(row.target) || 0) + 1);
  }));
  return [...seen.values()].some((n) => n > 1);
}

// ---------------------------------------------------------------------------
// run / poll
// ---------------------------------------------------------------------------
async function runJob() {
  let config;
  try {
    ({ config } = buildConfig({ strict: true }));
  } catch (error) {
    toast(error.message, "bad");
    return;
  }
  state.busy = true;
  $("#run-btn").disabled = true;
  $("#cancel-btn").hidden = false;
  $("#job-card").hidden = false;
  $("#log").textContent = "";
  $("#result").innerHTML = "";
  $("#job-state").textContent = "提交中";
  $("#job-state").className = "badge";
  state.logOffset = 0;
  try {
    const data = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({ kind: state.mode, config }),
    });
    state.job = data.job;
    $("#job-state").textContent = "运行中";
    startPolling();
  } catch (error) {
    state.busy = false;
    $("#run-btn").disabled = false;
    $("#cancel-btn").hidden = true;
    toast(error.message, "bad");
  }
}

function stopPolling() {
  if (state.pollTimer) clearTimeout(state.pollTimer);
  state.pollTimer = null;
}

function startPolling() {
  stopPolling();
  const tick = async () => {
    if (!state.job) return;
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(state.job.job_id)}?log_from=${state.logOffset}`);
      if (data.log) {
        const log = $("#log");
        log.textContent += data.log;
        log.scrollTop = log.scrollHeight;
        state.logOffset = data.log_offset;
      }
      const status = data.status || {};
      state.job = { ...state.job, ...status };
      const running = !["done", "error"].includes(status.state);
      setJobState(status);
      $("#progress-bar").style.width = `${progressFor(status.phase, status.state)}%`;
      if (data.result && state.lastResult !== data.result) {
        state.lastResult = data.result;
        renderResult(data.result, status);
      }
      if (running) {
        state.pollTimer = setTimeout(tick, 1000);
      } else {
        state.busy = false;
        $("#run-btn").disabled = false;
        $("#cancel-btn").hidden = true;
        refreshJobs();
        if (status.state === "error") toast(status.error || "任务失败", "bad");
        else toast("任务完成", "info");
      }
    } catch (error) {
      state.pollTimer = setTimeout(tick, 1800);
    }
  };
  tick();
}

function progressFor(phase, stateName) {
  if (stateName === "done") return 100;
  if (stateName === "error") return 100;
  const table = { starting: 4, build_bank: 22, train: 55, calibrate: 55, export: 88, done: 100 };
  return table[phase] ?? 12;
}

function setJobState(status) {
  const el = $("#job-state");
  const map = {
    running: ["运行中", "badge"], starting: ["启动中", "badge"],
    done: ["完成", "badge"], error: ["失败", "badge"],
  };
  const [text, cls] = map[status.state] || [status.state || "未知", "badge"];
  el.textContent = text;
  el.className = cls;
  if (status.state === "error") el.classList.add("bad");
}

async function cancelJob() {
  if (!state.job) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(state.job.job_id)}/cancel`, { method: "POST", body: "{}" });
    toast("已请求停止", "warn");
  } catch (error) {
    toast(error.message, "bad");
  }
}

async function loadJob(jobId) {
  if (!jobId) return;
  stopPolling();
  state.job = { job_id: jobId };
  state.logOffset = 0;
  state.lastResult = null;
  $("#job-card").hidden = false;
  $("#log").textContent = "";
  $("#result").innerHTML = "";
  try {
    const data = await api(`/api/jobs/${encodeURIComponent(jobId)}?log_from=0`);
    state.job = { job_id: jobId, ...(data.status || {}) };
    $("#log").textContent = data.log || "";
    state.logOffset = data.log_offset || 0;
    setJobState(data.status || {});
    if (data.result) { state.lastResult = data.result; renderResult(data.result, data.status || {}); }
    if (!["done", "error"].includes((data.status || {}).state)) startPolling();
  } catch (error) {
    toast(error.message, "bad");
  }
}

async function refreshJobs() {
  try {
    const data = await api("/api/jobs");
    renderJobs(data.jobs || []);
  } catch { /* ignore */ }
}

function renderJobs(jobs) {
  const sel = $("#job-history");
  const current = state.job?.job_id || "";
  sel.innerHTML = `<option value="">最近任务…</option>` + jobs.slice(0, 20).map((job) =>
    `<option value="${esc(job.job_id)}" ${job.job_id === current ? "selected" : ""}>${esc(job.job_id)} · ${esc(job.kind || "")} · ${esc(job.state || "")}</option>`).join("");
}

// ---------------------------------------------------------------------------
// result rendering
// ---------------------------------------------------------------------------
function fileLink(jobId, rel, label) {
  return `<a class="file-link" href="/files/${encodeURIComponent(jobId)}/${rel}">${esc(label)}</a>`;
}

function metricCard(label, value, suffix = "") {
  const text = value == null ? "N/A" : `${typeof value === "number" ? value.toFixed(4) : value}${suffix}`;
  return `<div class="metric"><div class="v">${esc(text)}</div><div class="l">${esc(label)}</div></div>`;
}

function recallRows(metrics, names) {
  return (metrics.per_class_recall || []).map((value, i) => {
    const support = metrics.support ? metrics.support[i] : null;
    const pct = value == null ? 0 : Math.max(0, Math.min(1, value)) * 100;
    return `<div class="recall"><span style="width:72px">${esc(names[i] || i)}</span>
      <div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div>
      <span style="width:58px;text-align:right">${value == null ? "N/A" : (value * 100).toFixed(1) + "%"}</span>
      <span class="muted" style="width:54px;text-align:right">n=${support == null ? "?" : support}</span></div>`;
  }).join("");
}

function confusionTable(metrics, names) {
  const matrix = metrics.confusion || [];
  if (!matrix.length) return "";
  const head = `<tr><th>真\\预</th>${names.map((n) => `<th>${esc(n)}</th>`).join("")}</tr>`;
  const body = matrix.map((row, i) => `<tr><th>${esc(names[i] || i)}</th>${row.map((v, j) =>
    `<td class="${i === j && v > 0 ? "hot" : ""}">${v}</td>`).join("")}</tr>`).join("");
  return `<table class="confusion">${head}${body}</table>`;
}

function renderResult(result, status) {
  const box = $("#result");
  const jobId = state.job?.job_id || "";
  const names = (result.outputs || []).map((o) => o.name);
  const links = [];
  if (result.model) links.push(fileLink(jobId, "model.pt", "模型 .pt"));
  if (result.onnx) {
    links.push(fileLink(jobId, "onnx/model.onnx", "ONNX"));
    links.push(fileLink(jobId, "onnx/manifest.json", "manifest"));
    links.push(fileLink(jobId, "onnx/verify_onnx.json", "ONNX 校验"));
  }
  links.push(fileLink(jobId, "report.json", "report.json"));

  if (result.kind === "source") {
    const evaluation = result.evaluation;
    let html = `<h3>源模型已生成</h3><div>${links.join("")}</div>`;
    html += `<div class="muted" style="margin-top:6px">${esc(result.model || "")}</div>`;
    if (evaluation) {
      const modeLabel = { trial: "会话内试次留出（诊断）", loso: "逐会话 LOSO",
                          holdout: "留出一份会话" }[evaluation.mode] || evaluation.mode;
      html += `<h3>留出评估 <span class="muted">（${esc(modeLabel)}）</span></h3>`;
      html += `<div class="grid2">${metricCard("平均 balanced accuracy",
        evaluation.mean_balanced_accuracy)}${metricCard("折数", evaluation.folds.length)}</div>`;
      html += `<table class="report"><thead><tr><th>留出</th><th>训练试次</th><th>测试试次</th><th>bacc</th><th>acc</th></tr></thead><tbody>`;
      evaluation.folds.forEach((fold) => {
        html += `<tr><td>${esc(fold.held_out)}</td><td>${fold.n_train_trials}</td>
          <td>${fold.n_test_trials}</td><td>${fold.metrics.balanced_accuracy == null ? "N/A" : fold.metrics.balanced_accuracy.toFixed(4)}</td>
          <td>${fold.metrics.accuracy == null ? "N/A" : fold.metrics.accuracy.toFixed(4)}</td></tr>`;
      });
      html += `</tbody></table>`;
      const first = evaluation.folds[0];
      if (first) {
        html += `<h3>第一折逐类召回</h3>${recallRows(first.metrics, names)}`;
        html += `<h3>混淆矩阵</h3>${confusionTable(first.metrics, names)}`;
      }
      html += `<p class="hint">${esc(evaluation.note || "")}</p>`;
    } else {
      html += `<p class="warn">没有留出评估数字。</p>`;
    }
    if (result.onnx) {
      const verify = result.onnx.verify;
      html += `<h3>ONNX 校验</h3><p class="${verify.pass_all ? "ok" : "bad"}">
        ${verify.pass_all ? "通过" : "未通过"} · ONNX vs numpy max|Δ| =
        ${verify.onnx_vs_numpy_max_abs_logit.toExponential(2)} · argmax 一致 =
        ${(verify.onnx_argmax_agreement * 100).toFixed(1)}%</p>`;
    }
    box.innerHTML = html;
    return;
  }

  if (result.kind === "calibration") {
    const metrics = result.metrics || {};
    const before = metrics.before || {};
    const after = metrics.after || {};
    let html = `<h3>校准模型已保存</h3><div>${links.join("")}</div>`;
    html += `<div class="muted" style="margin-top:6px">${esc(result.model || "")}</div>`;
    html += `<div class="grid2">${metricCard("校准前 bacc", before.balanced_accuracy)}
      ${metricCard("校准后 bacc", after.balanced_accuracy)}</div>`;
    html += `<p class="hint">${esc(metrics.note || "")}</p>`;
    html += `<table class="report"><thead><tr><th>集合</th><th>试次</th><th>bacc</th><th>acc</th></tr></thead><tbody>
      <tr><td>校准前</td><td>${before.n_trials ?? "—"}</td><td>${before.balanced_accuracy == null ? "N/A" : before.balanced_accuracy.toFixed(4)}</td><td>${before.accuracy == null ? "N/A" : before.accuracy.toFixed(4)}</td></tr>
      <tr><td>校准后</td><td>${after.n_trials ?? "—"}</td><td>${after.balanced_accuracy == null ? "N/A" : after.balanced_accuracy.toFixed(4)}</td><td>${after.accuracy == null ? "N/A" : after.accuracy.toFixed(4)}</td></tr>
      </tbody></table>`;
    html += `<h3>校准后逐类召回</h3>${recallRows(after, names)}`;
    html += `<h3>校准后混淆矩阵</h3>${confusionTable(after, names)}`;
    html += `<p class="hint">输出契约指纹 ${esc(result.output_fingerprint)}（与源模型一致） ·
      重载检查 max|Δ| = ${result.reload_check?.max_abs_logit_diff?.toExponential?.(2) ?? "0"}</p>`;
    box.innerHTML = html;
  }
}

// ---------------------------------------------------------------------------
// browse modal
// ---------------------------------------------------------------------------
async function openBrowser(mode) {
  state.browser.mode = mode;
  const start = mode === "model" && state.model ? state.model.path : $("#path-input").value.trim();
  try {
    const data = await api(`/api/browse?path=${encodeURIComponent(start || "")}`);
    state.browser.path = data.path;
    state.browser.parent = data.parent;
    state.browser.entries = data.entries || [];
    $("#modal-title").textContent = mode === "model" ? "选择模型 .pt" : "选择 Session 目录";
    $("#modal").hidden = false;
    renderBrowser();
  } catch (error) {
    toast(error.message, "bad");
  }
}

function renderBrowser() {
  $("#modal-path").textContent = state.browser.path;
  const list = $("#modal-list");
  const rows = state.browser.entries.filter((entry) =>
    entry.dir || (state.browser.mode === "model" && entry.name.endsWith(".pt")));
  list.innerHTML = rows.length ? rows.map((entry) => `
    <div class="entry" data-dir="${entry.dir ? 1 : 0}" data-path="${esc(entry.path)}">
      <span class="ico">${entry.dir ? "📁" : "🧠"}</span>
      <span class="name">${esc(entry.name)}</span>
      ${entry.kind ? `<span class="tag">${esc(KIND_LABEL[entry.kind] || entry.kind)}</span>` : ""}
    </div>`).join("") : `<div class="muted" style="padding:10px">空目录</div>`;
  list.querySelectorAll(".entry").forEach((entry) => {
    entry.addEventListener("click", async () => {
      const path = entry.dataset.path;
      if (entry.dataset.dir === "1") {
        try {
          const data = await api(`/api/browse?path=${encodeURIComponent(path)}`);
          state.browser.path = data.path;
          state.browser.parent = data.parent;
          state.browser.entries = data.entries || [];
          renderBrowser();
        } catch (error) { toast(error.message, "bad"); }
      } else {
        $("#model-input").value = path;
        $("#modal").hidden = true;
        await loadModel(path);
      }
    });
  });
  $("#modal-pick").textContent = state.browser.mode === "model" ? "使用这个目录？" : "选择此目录并扫描";
}

async function pickBrowser() {
  const path = state.browser.path;
  $("#modal").hidden = true;
  if (state.browser.mode === "model") {
    toast("请点选一个 .pt 文件", "warn");
    return;
  }
  $("#path-input").value = path;
  await scanPath(path);
}

function goUpBrowser() {
  if (!state.browser.parent) return;
  api(`/api/browse?path=${encodeURIComponent(state.browser.parent)}`).then((data) => {
    state.browser.path = data.path;
    state.browser.parent = data.parent;
    state.browser.entries = data.entries || [];
    renderBrowser();
  }).catch((error) => toast(error.message, "bad"));
}

// ---------------------------------------------------------------------------
// model list
// ---------------------------------------------------------------------------
async function openModelList() {
  try {
    const data = await api("/api/models");
    const models = data.models || [];
    $("#modal-title").textContent = "选择已有模型";
    $("#modal").hidden = false;
    $("#modal-path").textContent = "任务目录中的 .pt";
    $("#modal-list").innerHTML = models.length ? models.map((m) => `
      <div class="entry" data-model="${esc(m.path)}">
        <span class="ico">🧠</span><span class="name">${esc(m.name)}</span>
        <span class="tag">${esc(m.job_id)}</span>
      </div>`).join("") : `<div class="muted" style="padding:10px">还没有模型</div>`;
    $("#modal-list").querySelectorAll("[data-model]").forEach((entry) => {
      entry.addEventListener("click", async () => {
        $("#modal").hidden = true;
        $("#model-input").value = entry.dataset.model;
        await loadModel(entry.dataset.model);
      });
    });
    $("#modal-pick").textContent = "关闭";
  } catch (error) { toast(error.message, "bad"); }
}

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------
function bind() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
  $("#scan-btn").addEventListener("click", () => scanPath($("#path-input").value.trim()));
  $("#path-input").addEventListener("keydown", (e) => { if (e.key === "Enter") scanPath($("#path-input").value.trim()); });
  $("#browse-btn").addEventListener("click", () => openBrowser("data"));
  $("#confirm-all").addEventListener("click", confirmAllSources);
  $("#model-btn").addEventListener("click", () => loadModel($("#model-input").value.trim()));
  $("#model-input").addEventListener("keydown", (e) => { if (e.key === "Enter") loadModel($("#model-input").value.trim()); });
  $("#model-list-btn").addEventListener("click", openModelList);
  $("#outputs-confirm").addEventListener("change", () => {
    state.outputsConfirmed = $("#outputs-confirm").checked;
    updatePreview();
  });
  $("#preset").addEventListener("change", () => {
    const presets = {
      quick: [20, 10, 20, 10],
      standard: [200, 30, 200, 40],
      long: [300, 40, 300, 60],
    };
    const value = $("#preset").value;
    if (presets[value]) {
      $("#pre-epochs").value = presets[value][0];
      $("#head-epochs").value = presets[value][1];
      $("#ft-epochs").value = presets[value][2];
      $("#patience").value = presets[value][3];
    }
    updatePreview();
  });
  ["#profile", "#device", "#seed", "#job-name", "#pre-epochs", "#head-epochs", "#ft-epochs",
   "#patience", "#batch", "#eval-mode", "#final-fit", "#export-onnx-train", "#cal-c",
   "#cal-holdout", "#export-onnx-cal", "#many-confirm", "#legacy-profile-confirm"].forEach((sel) => {
    const element = $(sel);
    if (element) element.addEventListener("change", updatePreview);
    if (element && element.type === "number") element.addEventListener("input", updatePreview);
  });
  $("#run-btn").addEventListener("click", runJob);
  $("#cancel-btn").addEventListener("click", cancelJob);
  $("#job-history").addEventListener("change", (e) => loadJob(e.target.value));
  $("#modal-close").addEventListener("click", () => { $("#modal").hidden = true; });
  $("#modal-up").addEventListener("click", goUpBrowser);
  $("#modal-pick").addEventListener("click", () => {
    if (state.browser.mode === "model") { $("#modal").hidden = true; return; }
    pickBrowser();
  });
  $("#modal").addEventListener("click", (e) => { if (e.target === $("#modal")) $("#modal").hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#modal").hidden = true; });
}

bind();
boot();
