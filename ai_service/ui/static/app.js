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
const MLUI_CLINIC_STORAGE_KEY = "vetai_mlops_clinic_id";

function parseClinicFromUrl() {
  const p = new URLSearchParams(window.location.search || "");
  const v = (p.get("clinic_id") || p.get("clinic") || "").trim();
  return v || null;
}

/** Keep URL and localStorage in sync with the dropdown — bookmark /mlops-ui?clinic_id=... */
function syncClinicToUrlAndStorage(clinicKey) {
  const u = new URL(window.location.href);
  if (clinicKey) {
    u.searchParams.set("clinic_id", clinicKey);
    u.searchParams.delete("clinic");
  } else {
    u.searchParams.delete("clinic_id");
    u.searchParams.delete("clinic");
  }
  const next = u.pathname + u.search + u.hash;
  if (next !== window.location.pathname + window.location.search + window.location.hash) {
    history.replaceState(null, "", next);
  }
  try {
    if (clinicKey) localStorage.setItem(MLUI_CLINIC_STORAGE_KEY, clinicKey);
    else localStorage.removeItem(MLUI_CLINIC_STORAGE_KEY);
  } catch (_) {
    /* ignore */
  }
}

function updateClinicContextBanner() {
  const label = document.getElementById("clinic-context-label");
  const sel = document.getElementById("clinic-scope");
  if (!label || !sel) return;
  const key = getSelectedClinicKey();
  if (!key) {
    label.textContent = "global (shared)";
    return;
  }
  const opt = sel.options[sel.selectedIndex];
  label.textContent = opt && opt.textContent ? opt.textContent.trim() : key;
}

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

/** Multipart POST (do not set Content-Type — browser sets boundary). */
async function jpostMultipart(url, formData) {
  const r = await fetch(url, {
    method: "POST",
    headers: { Accept: "application/json", ...authHeaders() },
    body: formData
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

/** Selected clinic id from customers-service (UUID string or legacy numeric string), or null for global. */
function getSelectedClinicKey() {
  const el = document.getElementById("clinic-scope");
  if (!el) return null;
  const v = (el.value || "").trim();
  return v || null;
}

function clinicQuerySuffix() {
  const cid = getSelectedClinicKey();
  return cid == null ? "" : `?clinic_id=${encodeURIComponent(cid)}`;
}

function clinicQueryPrefix() {
  const cid = getSelectedClinicKey();
  return cid == null ? "" : `&clinic_id=${encodeURIComponent(cid)}`;
}

async function loadClinicOptions() {
  const sel = document.getElementById("clinic-scope");
  if (!sel) return;
  const prev = sel.value;
  const data = await jget("/mlops/clinics");
  const srcEl = document.getElementById("clinics-source");
  if (srcEl && data.source) {
    srcEl.textContent =
      data.source === "customers-service"
        ? "· live"
        : data.source === "stale"
          ? "· cached (customers down)"
          : data.source === "error"
            ? "· error (no clinics)"
            : `· ${data.source}`;
  }
  sel.innerHTML = `<option value="">Global (shared default)</option>`;
  (data.clinics || []).forEach((c) => {
    const opt = document.createElement("option");
    opt.value = String(c.id);
    opt.textContent = `${c.name} (${c.id})`;
    sel.appendChild(opt);
  });

  let fromUrl = parseClinicFromUrl();
  let fromStore = null;
  try {
    fromStore = (localStorage.getItem(MLUI_CLINIC_STORAGE_KEY) || "").trim() || null;
  } catch (_) {
    fromStore = null;
  }
  const pick = [fromUrl, fromStore, prev].find(
    (k) => k && [...sel.options].some((o) => o.value === k)
  );
  if (pick) sel.value = pick;

  syncClinicToUrlAndStorage(getSelectedClinicKey());
  updateClinicContextBanner();
}

async function refreshModel() {
  const cid = getSelectedClinicKey();
  const q = cid == null ? "" : `?clinic_id=${encodeURIComponent(cid)}`;
  const models = await jget("/mlops/models" + q);

  const hint = document.getElementById("scope-hint");
  const activeLabel = document.getElementById("active-label");
  if (models.scope === "clinic") {
    if (hint) {
      hint.textContent =
        models.effective_source === "clinic_pin"
          ? `Clinic ${models.clinic_id}: using pinned model. Global default: ${models.global_active ?? "—"}.`
          : `Clinic ${models.clinic_id}: no pin — using global default (${models.global_active ?? "—"}).`;
    }
    if (activeLabel) activeLabel.textContent = "Effective model";
    setText("active-model", models.effective || "-");
  } else {
    if (hint) {
      hint.textContent =
        "Managing the global default model (used when a clinic has no pinned override).";
    }
    if (activeLabel) activeLabel.textContent = "Active";
    setText("active-model", models.active || "-");
  }

  const compareVersion =
    models.scope === "clinic" ? models.effective : models.active;
  const sel = document.getElementById("model-select");
  sel.innerHTML = "";
  (models.versions || []).forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    if (v === compareVersion) opt.selected = true;
    sel.appendChild(opt);
  });

  const info = await jget("/model/info");
  const merged = {
    ...info,
    mlops_scope: models.scope,
    clinic_id: models.clinic_id ?? null,
    pinned_version: models.pinned ?? null,
    effective_version: models.effective ?? null,
    effective_source: models.effective_source ?? null,
    global_active: models.global_active ?? null,
  };
  setPre("model-info", merged);
}

async function refreshTraining() {
  const cid = getSelectedClinicKey();
  const q = clinicQuerySuffix();
  const el = await jget("/continuous-training/training/eligibility" + q);
  setText("eligibility", `${el.eligible_feedback_count}/${el.training_threshold} eligible`);

  const scopeEl = document.getElementById("training-scope-label");
  if (scopeEl) {
    scopeEl.textContent =
      cid == null
        ? "Scope: global (all clinics — eligibility counts every clinic’s feedback)."
        : `Scope: this clinic only (eligibility & last job are filtered by clinic_id).`;
  }

  const last = await jget("/mlops/training/last" + q);
  const lj = last.last_training;
  if (!lj) {
    setText("last-job", "none");
  } else {
    setText("last-job", `#${lj.training_id} ${lj.status} → ${lj.new_model_version || "-"}`);
  }

  const alignEl = document.getElementById("mlflow-align");
  if (alignEl) {
    try {
      const mv = await jget("/mlops/mlflow/latest-vs-active" + q);
      const lr = mv.latest_run;
      const latestVer = lr && lr.inferred_model_version ? lr.inferred_model_version : "—";
      const activeVer = mv.active_model_version != null ? mv.active_model_version : "—";
      let line;
      if (mv.versions_match === true) {
        line = `Matched: active = MLflow = ${activeVer}`;
      } else if (mv.versions_match === false) {
        line = `Mismatch: active=${activeVer} | latest MLflow=${latestVer}`;
      } else if (mv.status === "no_experiment" || mv.status === "no_runs") {
        line = `${mv.status}: ${mv.note || "—"}`;
      } else if (mv.status === "mlflow_error") {
        line = `MLflow error: ${mv.note || "—"}`;
      } else {
        line = mv.note || `active=${activeVer} | MLflow=${latestVer} | run=${lr ? lr.run_id : "—"}`;
      }
      alignEl.textContent = line;
    } catch (e) {
      alignEl.textContent =
        "Cannot read /mlops/mlflow/latest-vs-active (MLflow is down or network failed).";
    }
  }
}

async function loadHistory() {
  const offset = (historyPage - 1) * HISTORY_PAGE_SIZE;
  const cq = getSelectedClinicKey() == null ? "" : clinicQueryPrefix();
  const h = await jget(
    `/continuous-training/training/history?limit=${HISTORY_PAGE_SIZE}&offset=${offset}${cq}`
  );
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
    const clinicCell =
      run.clinic_id != null && run.clinic_id !== ""
        ? run.clinic_id
        : "—";
    tr.innerHTML = `
      <td>${run.training_id}</td>
      <td>${clinicCell}</td>
      <td>${statusBadge}</td>
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
  const cid = getSelectedClinicKey();
  const q =
    cid == null
      ? "?days=7"
      : `?days=7&clinic_id=${encodeURIComponent(cid)}`;
  const d = await jget("/mlops/drift/summary" + q);
  setPre("drift-summary", d);
}

async function loadRegistryStatusV2() {
  const d = await jget("/mlops/v2/registry/status");
  setPre("registry-status", d);
}

async function loadArtifactStorage() {
  const d = await jget("/mlops/artifact-storage");
  setPre("artifact-storage", d);
}

async function refreshAll() {
  await refreshModel();
  await refreshTraining();
  await loadHistory();
  await loadPolicy();
  await loadDrift();
  await loadRegistryStatusV2();
  await loadArtifactStorage();
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
    try {
      await refreshAll();
    } catch (e) {
      toast(`Refresh failed (${e.status || "?"})`);
    }
  });
  document.getElementById("clinic-scope").addEventListener("change", async () => {
    historyPage = 1;
    syncClinicToUrlAndStorage(getSelectedClinicKey());
    updateClinicContextBanner();
    try {
      await refreshAll();
    } catch (e) {
      toast(`Refresh failed (${e.status || "?"})`);
    }
  });
  document.getElementById("btn-set-active").addEventListener("click", async () => {
    const mv = document.getElementById("model-select").value;
    const cid = getSelectedClinicKey();
    try {
      let r;
      if (cid == null) {
        r = await jsend("/models/active", "POST", { model_version: mv });
        toast(`Global active model set: ${r.active}`);
      } else {
        r = await jsend(`/models/clinic/${encodeURIComponent(cid)}/active`, "POST", { model_version: mv });
        toast(`Clinic ${cid} pinned to: ${r.model_version || mv}`);
      }
      await refreshModel();
    } catch (e) {
      toast(`Set active failed (${e.status || "?"}): ${JSON.stringify(e.data)}`);
    }
  });
  document.getElementById("btn-trigger").addEventListener("click", async () => {
    const t = document.getElementById("trigger-type").value;
    try {
      const tcid = getSelectedClinicKey();
      const body = {
        trigger_type: t,
        force: false,
        training_mode: "local"
      };
      if (tcid != null) body.clinic_id = tcid;
      const r = await jsend("/continuous-training/training/trigger", "POST", body);
      toast(`Triggered training #${r.training_id}`);
      await refreshTraining();
      await loadHistory();
    } catch (e) {
      toast(`Trigger failed (${e.status || "?"}): ${JSON.stringify(e.data)}`);
    }
  });
  document.getElementById("btn-bootstrap-csv")?.addEventListener("click", async () => {
    if (!getToken()) {
      toast("Admin token required in the navbar (same as server ADMIN_TOKEN).");
      return;
    }
    const input = document.getElementById("bootstrap-csv-file");
    if (!input?.files?.length) {
      toast("Choose a CSV file first.");
      return;
    }
    const fd = new FormData();
    fd.append("file", input.files[0]);
    const tcid = getSelectedClinicKey();
    if (tcid != null) fd.append("clinic_id", tcid);
    fd.append("training_mode", "local");
    try {
      const r = await jpostMultipart("/continuous-training/training/bootstrap-csv", fd);
      toast(`Bootstrap training #${r.training_id} started (${r.row_count} rows, scope=${r.training_scope || "?"})`);
      await refreshTraining();
      await loadHistory();
    } catch (e) {
      const d = e.data?.detail ?? e.data ?? e;
      toast(`Bootstrap failed (${e.status || "?"}): ${typeof d === "string" ? d : JSON.stringify(d)}`);
    }
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
  document.getElementById("btn-save-policy").addEventListener("click", async () => {
    try {
      await savePolicy();
    } catch (e) {
      const d = e.data?.detail ?? e.data ?? e;
      const msg = typeof d === "string" ? d : JSON.stringify(d);
      toast(`Save policy failed (${e.status || "?"}): ${msg}`);
    }
  });

  try {
    await loadClinicOptions();
    await refreshAll();
  } catch (e) {
    toast(`Initial load failed (${e.status || "?"})`);
  }
}

main();

