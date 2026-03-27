async function jget(url) {
  const r = await fetch(url, { headers: { "Accept": "application/json" } });
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
  if (!r.ok) throw { status: r.status, data };
  return data;
}

const HISTORY_PAGE_SIZE = 10;
let historyPage = 1;

function getToken() {
  const fromInput = (document.getElementById("admin-token")?.value || "").trim();
  const fromStorage = (localStorage.getItem("vetai_admin_token") || "").trim();
  const v = fromInput || fromStorage;
  return v || null;
}

function authHeaders() {
  const t = getToken();
  return t ? { "Authorization": `Bearer ${t}` } : {};
}

async function jsend(url, method, body) {
  const r = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json", "Accept": "application/json", ...authHeaders() },
    body: JSON.stringify(body || {})
  });
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (_) { data = text; }
  if (!r.ok) throw { status: r.status, data };
  return data;
}

async function downloadWithAuth(url, filename) {
  const r = await fetch(url, { headers: { ...authHeaders() } });
  if (!r.ok) {
    const text = await r.text();
    throw { status: r.status, data: text };
  }
  const blob = await r.blob();
  const a = document.createElement("a");
  const objUrl = URL.createObjectURL(blob);
  a.href = objUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objUrl);
}

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v == null ? "-" : String(v);
}

function setPre(id, obj) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = obj == null ? "-" : JSON.stringify(obj, null, 2);
}

function toast(msg) {
  alert(msg);
}

async function refreshModel() {
  const models = await jget("/mlops/models");
  setText("active-model", models.active || "-");

  const sel = document.getElementById("model-select");
  sel.innerHTML = "";
  (models.versions || []).forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === models.active) opt.selected = true;
    sel.appendChild(opt);
  });

  const info = await jget("/model/info");
  setPre("model-info", info);
}

async function refreshTraining() {
  const el = await jget("/continuous-training/training/eligibility");
  setText("eligibility", `${el.eligible_feedback_count}/${el.training_threshold} eligible`);

  const last = await jget("/mlops/training/last");
  const lj = last.last_training;
  if (!lj) {
    setText("last-job", "none");
  } else {
    setText("last-job", `#${lj.training_id} ${lj.status} → ${lj.new_model_version || "-"}`);
  }
}

async function loadHistory() {
  const offset = (historyPage - 1) * HISTORY_PAGE_SIZE;
  const h = await jget(`/continuous-training/training/history?limit=${HISTORY_PAGE_SIZE}&offset=${offset}`);
  const tbody = document.getElementById("history-body");
  const pag = document.getElementById("history-pagination");
  tbody.innerHTML = "";
  (h.training_runs || []).forEach(run => {
    const tr = document.createElement("tr");
    const statusBadge = run.status === "completed"
      ? `<span class="badge text-bg-success">${run.status}</span>`
      : run.status === "failed"
        ? `<span class="badge text-bg-danger">${run.status}</span>`
        : `<span class="badge text-bg-secondary">${run.status || "-"}</span>`;
    tr.innerHTML = `
      <td>${run.training_id}</td>
      <td>${statusBadge}</td>
      <td>${run.training_mode || "-"}</td>
      <td class="mono">${run.new_model_version || "-"}</td>
      <td>${run.validation_accuracy ?? "-"}</td>
      <td>${run.f1_score ?? "-"}</td>
      <td>${run.dataset_row_count ?? 0}</td>
      <td>
        <button class="btn btn-sm btn-outline-secondary btn-status" data-id="${run.training_id}">Status</button>
        <button class="btn btn-sm btn-outline-primary btn-download-data" data-id="${run.training_id}">Download data</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
  if (!h.training_runs || h.training_runs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-muted">No runs</td></tr>`;
  }

  const total = Number(h.total_count || 0);
  const pages = Math.max(1, Math.ceil(total / HISTORY_PAGE_SIZE));
  pag.innerHTML = "";
  for (let i = 1; i <= pages; i++) {
    const li = document.createElement("li");
    li.className = `page-item ${i === historyPage ? "active" : ""}`;
    li.innerHTML = `<button class="page-link history-page-btn" data-page="${i}">${i}</button>`;
    pag.appendChild(li);
  }
}

async function loadPolicy() {
  const p = await jget("/continuous-training/config");
  document.getElementById("policy-threshold").value = p.training_threshold;
  document.getElementById("policy-window").value = p.training_window_days;
}

async function savePolicy() {
  if (!getToken()) {
    toast(
      "Admin token required: enter the same value as server ADMIN_TOKEN in the navbar (top right), then Save policy again."
    );
    return;
  }
  const threshold = Number(document.getElementById("policy-threshold").value);
  const windowDays = Number(document.getElementById("policy-window").value);
  const r = await jsend("/continuous-training/config", "PUT", {
    training_threshold: threshold,
    training_window_days: windowDays
  });
  toast(`Saved policy: threshold=${r.training_threshold}, window=${r.training_window_days}`);
  await refreshTraining();
}

async function loadDrift() {
  const d = await jget("/mlops/drift/summary?days=7");
  setPre("drift-summary", d);
}

async function main() {
  const tokenEl = document.getElementById("admin-token");
  if (tokenEl) {
    tokenEl.value = localStorage.getItem("vetai_admin_token") || "";
    const persistToken = () => {
      const t = tokenEl.value.trim();
      if (t) localStorage.setItem("vetai_admin_token", t);
    };
    tokenEl.addEventListener("input", persistToken);
    tokenEl.addEventListener("change", persistToken);
  }
  document.getElementById("btn-refresh").addEventListener("click", async () => {
    try { await refreshModel(); await refreshTraining(); } catch (e) { toast(`Refresh failed (${e.status || "?"})`); }
  });
  document.getElementById("btn-set-active").addEventListener("click", async () => {
    const mv = document.getElementById("model-select").value;
    try {
      const r = await jsend("/models/active", "POST", { model_version: mv });
      toast(`Active model set: ${r.active}`);
      await refreshModel();
    } catch (e) {
      toast(`Set active failed (${e.status || "?"}): ${JSON.stringify(e.data)}`);
    }
  });
  document.getElementById("btn-trigger").addEventListener("click", async () => {
    const t = document.getElementById("trigger-type").value;
    try {
      const r = await jsend("/continuous-training/training/trigger", "POST", {
        trigger_type: t,
        force: false,
        training_mode: "local"
      });
      toast(`Triggered training #${r.training_id}`);
      await refreshTraining();
      await loadHistory();
    } catch (e) {
      toast(`Trigger failed (${e.status || "?"}): ${JSON.stringify(e.data)}`);
    }
  });
  document.getElementById("btn-load-history").addEventListener("click", async () => {
    try { await loadHistory(); } catch (e) { toast(`History failed (${e.status || "?"})`); }
  });
  document.getElementById("history-pagination").addEventListener("click", async (ev) => {
    const btn = ev.target.closest(".history-page-btn");
    if (!btn) return;
    const p = Number(btn.getAttribute("data-page") || "1");
    if (!Number.isFinite(p) || p < 1) return;
    historyPage = p;
    try { await loadHistory(); } catch (e) { toast(`History page failed (${e.status || "?"})`); }
  });
  document.getElementById("history-body").addEventListener("click", async (ev) => {
    const statusBtn = ev.target.closest(".btn-status");
    if (statusBtn) {
      const id = statusBtn.getAttribute("data-id");
      try {
        const data = await jget(`/continuous-training/training/status?training_id=${encodeURIComponent(id)}`);
        toast(JSON.stringify(data, null, 2));
      } catch (e) {
        toast(`Load status failed (${e.status || "?"}): ${JSON.stringify(e.data)}`);
      }
      return;
    }
    const dlBtn = ev.target.closest(".btn-download-data");
    if (dlBtn) {
      if (!getToken()) {
        toast("Admin token required to download training data.");
        return;
      }
      const id = dlBtn.getAttribute("data-id");
      try {
        await downloadWithAuth(
          `/continuous-training/training/dataset/download?training_id=${encodeURIComponent(id)}`,
          `training_${id}_dataset.csv`
        );
      } catch (e) {
        toast(`Download failed (${e.status || "?"}): ${JSON.stringify(e.data)}`);
      }
    }
  });
  document.getElementById("btn-load-policy").addEventListener("click", async () => {
    try { await loadPolicy(); } catch (e) { toast(`Load policy failed (${e.status || "?"})`); }
  });
  document.getElementById("btn-save-policy").addEventListener("click", async () => {
    try {
      await savePolicy();
    } catch (e) {
      const d = e.data?.detail ?? e.data ?? e;
      const msg = typeof d === "string" ? d : JSON.stringify(d);
      toast(`Save policy failed (${e.status || "?"}): ${msg}`);
    }
  });
  document.getElementById("btn-drift").addEventListener("click", async () => {
    try { await loadDrift(); } catch (e) { toast(`Drift failed (${e.status || "?"})`); }
  });

  try {
    await refreshModel();
    await refreshTraining();
    await loadPolicy();
  } catch (e) {
    toast(`Initial load failed (${e.status || "?"})`);
  }
}

main();

