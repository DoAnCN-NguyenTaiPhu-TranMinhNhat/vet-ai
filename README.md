# Vet-AI

FastAPI service for **veterinary diagnosis** (sklearn RandomForest), **continuous learning** from clinician feedback, **MLOps** (MLflow, optional S3 artifacts), and **Prometheus** metrics.

---

## Overview

Vet-AI exposes a REST API consumed by the Spring **genai-service** (and similar clients). It loads trained models from disk, runs inference on structured visit features, logs predictions and feedback to **PostgreSQL** for retraining, and can promote models per clinic (multi-tenant) or globally.

| Topic | Description |
|--------|-------------|
| Inference | `POST /predict` — symptoms and visit features → ranked diagnoses |
| Feedback | `POST /continuous-training/feedback` — accept/reject + final diagnosis for training |
| Training | In-process or **EKS/K8s** job (`scripts/train_eks.py`) — see `docker/Dockerfile.training` |
| Registry | Active model pins, MLflow runs, optional champion/challenger under `/mlops/v2` |

OpenAPI docs: **`GET /docs`** (Swagger UI) when the app is running.

---

## Repository layout

```text
vet-ai/
├── ai_service/               # FastAPI app + MLOps modules
├── docker/                   # Compose manifests (dev/prod)
├── docs/                     # Architecture, deployment, local dev docs
├── env/                      # Environment templates
├── scripts/                  # Training and helper scripts
├── monitoring/               # Monitoring assets
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.training
└── pyproject.toml
```

- Dev compose: `docker/compose.dev.yml`
- Prod compose: `docker/compose.prod.yml`
- Env template: `env/.env.example`

---

## Features

- **Inference** with per-request latency and confidence; optional **clinic-scoped** active model.
- **Continuous training**: policy thresholds, job history, CSV bootstrap (admin), feedback pool (GLOBAL vs CLINIC_ONLY).
- **Postgres** persistence for predictions, feedback, and training jobs (`DATABASE_URL`).
- **MLflow** experiment tracking (`MLFLOW_TRACKING_URI`).
- **Prometheus**: `/metrics` (via `prometheus-fastapi-instrumentator`) plus custom counters/histograms in `ai_service/metrics.py`.

---

## Related repositories

| Repository | Role |
|------------|------|
| [vet-microservices](https://github.com/DoAnCN-NguyenTaiPhu-TranMinhNhat/vet-microservices) | Spring Cloud stack; **genai-service** proxies diagnosis and training to Vet-AI (`VETAI_DIAGNOSIS_URL`). |
| [vet-microservices-config](https://github.com/DoAnCN-NguyenTaiPhu-TranMinhNhat/vet-microservices-config) | Central Spring configuration for the Java services (not for Vet-AI itself). |

Typical local layout (sibling folders):

```text
DACN/
├── vet-ai/                    # this repo
├── vet-microservices/         # docker-compose builds vet-ai from ../vet-ai
├── vet-microservices-config/
└── vet-infra/                 # optional: .env shared with compose
```

---

## Requirements

- **Python 3.11** (see `docker/Dockerfile.api`)
- **PostgreSQL** for continuous training / feedback (same DB can be shared with MLflow in dev; production often uses separate DB/schema)
- **MLflow** server (optional but recommended for experiment tracking)
- **Docker** or **Podman** (optional) — images: `docker/Dockerfile.api`, `docker/Dockerfile.training`

Install dependencies:

```bash
cd vet-ai
pip install -r requirements.txt
# Ensure python-multipart is installed (FastAPI forms)
pip install "python-multipart>=0.0.20"
```

---

## Configuration (environment variables)

### Environment files in this repository

| File | Purpose | Loaded automatically? |
|------|---------|---------------------|
| **`env/.env.example`** | **Committed template.** Full variable set and defaults. | No — template only |
| **`.env.local`** | **Local runtime file** (gitignored). Use this for local `uvicorn` / scripts. | No — load with `source` or inject in your shell |

In Docker, **`vet-microservices`** Compose usually injects variables via **`../vet-infra/.env`** for the `vet-ai` service — not from these files unless you wire them yourself.

Start from **`env/.env.example`** in this repo. Copy values into **`.env.local`** for local runs, or merge needed keys into **`vet-infra/.env`** when using `vet-microservices` Compose.

Below are the most important variables. Training and ML code honor many more (split ratios, gates, fine-tuning) — see `ai_service/training_engine.py` and `ai_service/continuous_training.py`.

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | PostgreSQL DSN for predictions, feedback, training jobs (required for full continuous training). |
| `ADMIN_TOKEN` | Bearer token for admin routes (`Authorization: Bearer <token>`). |
| `MODEL_DIR` / `MODEL_ROOT_DIR` | Paths to loaded model artifacts (`model.pkl`, encoders). |
| `MLFLOW_TRACKING_URI` | MLflow server URL (e.g. `http://mlflow:5000`). |
| `MLFLOW_EXPERIMENT_NAME` | Experiment name (default `vet-ai-continuous-training`). |
| `TRAINING_WINDOW_DAYS` | Rolling window for eligible feedback (days). |
| `ALLOW_EKS_HYBRID_TRAINING` | Set to `true` to run training as a separate K8s job from the API (advanced). |

JWT / external LLM keys are **not** required for core Vet-AI behavior; those belong to optional Java **GenAI chat** features in `vet-microservices`, not this service.

---

## Docker

**API image** (`docker/Dockerfile.api`):

```bash
docker build -t vet-ai:latest -f docker/Dockerfile.api .
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e ADMIN_TOKEN=... \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  vet-ai:latest
```

**Training worker image** (`docker/Dockerfile.training`) — used for CPU-heavy jobs; entrypoint scripts include `scripts/train_eks.py` / local training helpers.

In **vet-microservices**, `docker-compose.yml` usually builds this repo as service `vet-ai` and sets env from `../vet-infra/.env`. See that repository’s README for ports and dependencies.

---

## API map (summary)

| Prefix / path | Description |
|----------------|-------------|
| `POST /predict` | Main diagnosis inference. |
| `GET /health`, `/readyz`, `/livez` | Probes. |
| `GET /model/info`, `/models/versions` | Loaded model metadata. |
| `POST /models/active`, `/models/clinic/{id}/active` | Switch active model (admin / internal use). |
| `/continuous-training/*` | Config, eligibility, feedback, prediction logging, training trigger/history, bootstrap CSV (admin). |
| `/mlops/*` | Registry, drift, alignment helpers. |
| `/mlops/v2/*` | Champion–challenger workflow (Bearer admin). |
| `/mlair/status`, `/mlair/runs/training`, `/mlair/runs/{run_id}`, `POST /mlair/models/sync`, `POST /mlair/models/dedupe-versions` | Optional MLAir bridge: sync registry, dedupe duplicate version rows (admin token). |
| `POST /internal/mlair/model-promotion` | Inbound MLAir promote webhook (Bearer); behavior controlled by `VETAI_MLAIR_PROMOTION_MODE`. |
| `/metrics` | Prometheus scrape endpoint. |

---

## MLAir integration (optional)

Vet-AI can trigger and inspect MLAir runs via built-in bridge endpoints:

- `GET /mlair/status`
- `POST /mlair/runs/training` (admin token required)
- `GET /mlair/runs/{run_id}`

Required environment variables (see `env/.env.example`):

- `MLAIR_ENABLED=true`
- `MLAIR_API_BASE_URL` (for example `http://ml-air-api:8080` in Docker network)
- `MLAIR_TENANT_ID`, `MLAIR_PROJECT_ID`, `MLAIR_PIPELINE_ID`
- `MLAIR_AUTH_TOKEN` (if MLAir API requires bearer auth)

Full schemas: **`/docs`**.

### Scope behavior (tenant/project)

For multi-tenant setups, Vet-AI can sync MLAir scopes per clinic.

Recommended defaults:

- `MLAIR_MODEL_SCOPE_PER_CLINIC=true`
- `MLAIR_ENSURE_CLINIC_SCOPES=true`
- `MLAIR_ENSURE_CLINIC_TRAINING_PIPELINES=true` (creates an initial MLAir pipeline version per `clinic_*` project on sync so training can use the latest DAG)

Pipeline tasks created by Vet-AI use MLAir reference plugin names **`app_etl_adapter`** and **`app_train_adapter`**. The MLAir **API** image must include those plugins via the `mlair.plugins` entry-point group (the `ml-air` repo ships `mlair-reference-plugins` in the API Docker build). If you see `Train blocked (PLUGIN_NOT_FOUND)`, rebuild or upgrade the **`ml-air-api`** image you run against, then call `POST /mlair/models/sync` again.

Default scope mapping:

- Global model scope -> `project_id=default_project`
- Clinic model scope -> `project_id=clinic_<clinic_id_slug>`

**`VETAI_MLAIR_MIRROR_GLOBAL_VERSIONS_IN_CLINIC_PROJECTS`** (default `true`): when `POST /mlair/models/sync` runs, Vet-AI also registers, for **each clinic in the catalog**, the same merged list of versions as inference uses (`list_user_visible_model_versions` for that clinic — global + clinic disk). Those rows land in that clinic’s MLAir `clinic_*` project (same `file://` URIs as global where applicable), so the MLAir UI for clinic A aligns with Vet-AI. **Training started for clinic A** still saves new weights only under `models/clinics/<slug>/` and is scoped to clinic A in MLAir (not “global-only” output).

If MLAir shows **many** lines for one clinic, that is usually **one logical model** (`vet-<clinic>`) with **one registry row per distinct weights folder** (each `v*` on disk). Symlinks that point at the same folder are deduped by **canonical `file://` + realpath**. To cap how many merged versions are pushed per clinic, set **`VETAI_MLAIR_MIRROR_CLINIC_MAX_VERSIONS`** (e.g. `5`); `0` means no cap.

**Duplicate versions after each sync:** if the MLAir API only returned the first page of versions, Vet-AI could not see existing `artifact_uri` rows and would POST again. Sync now uses **paginated** `GET .../models/{id}/versions` (`_list_model_versions_all_items`, tunable via **`MLAIR_MODEL_VERSIONS_PAGE_SIZE`** / **`MLAIR_MODEL_VERSIONS_MAX_PAGES`**). Older duplicate rows in MLAir must be removed or archived in MLAir if you want a clean count.

**Dedupe tool:** `POST /mlair/models/dedupe-versions` (admin Bearer) groups MLAir rows by canonical `file://` realpath, keeps the best **stage** (production > staging > other) then highest **version** int, and **DELETE**s the rest (`DELETE .../models/{model_id}/versions/{version}` — requires your MLAir build to support that route). Query: `dry_run=true` (default) preview; `dry_run=false` apply. `clinic_id=<uuid>` one clinic; `all_scopes=true` global + catalog clinics; omit both for **global** project only.

If MLAir UI dropdown only shows `global/all`, verify:

1. Vet-AI has run model sync: `POST /mlair/models/sync`
2. MLAir `/v1/tenants/{tenant}/projects` returns clinic projects
3. Token used in MLAir UI has access to the target tenant/projects

### Model promotion webhook (MLAir → Vet-AI)

When MLAir is configured with **`MLAIR_MODEL_PROMOTE_WEBHOOK_URL`** pointing at this service, it calls:

- **`POST /internal/mlair/model-promotion`** (Bearer: `MLAIR_MODEL_PROMOTE_WEBHOOK_BEARER_TOKEN` on Vet-AI, same secret MLAir uses to sign the request).

**`VETAI_MLAIR_PROMOTION_MODE`** controls whether Vet-AI creates extra on-disk entries:

| Mode | Behavior |
|------|----------|
| **`materialize`** (default) | Map MLAir’s `file://` artifact into `MODEL_ROOT` (symlink or copy). If the same weights already exist as a normal `v*` folder (same realpath), **reuse that name** — no extra `v_mlair_*`. Internal `v_mlair_*` aliases (when created) are **not listed** in user-facing model APIs (`list_user_visible_model_versions` / `GET /predict/models`). |
| **`reuse_only`** | **No new files.** Resolve `artifact_uri` to a real path and **only** call `set_clinic_active_model` / global active if that path **already equals** an existing `v*` folder under `MODEL_ROOT` (e.g. output of continuous training). If nothing matches → **409** — deploy weights into a `v*` directory first, or temporarily use `materialize`. |

The external MLAir worker’s finetune path (`plugin_context.artifact_uri`) honors the same flag: in **`reuse_only`**, it reuses an existing matching `v*` folder instead of creating `v_mlair_*_u…` when possible.

**MLAir does not need source changes** for either mode; only Vet-AI env and (optionally) whether the webhook URL is set.

After you change the clinic/global **active** pin (UI, `POST /models/clinic/{id}/active`, or a successful promotion), **`VETAI_MLAIR_SYNC_ON_ACTIVE_CHANGE`** (default `true`) refreshes MLAir model stages so registry **production** tracks the same artifact as Vet-AI’s active model.

### External MLAir worker (optional)

When MLAir runs with **`ML_AIR_TASK_EXECUTION_MODE=external`**, Vet-AI can **lease** tasks, run **continuous training** for `app_train_adapter`, and **complete** runs with real metrics (HTTP to this service’s `/continuous-training/...` using `ADMIN_TOKEN`).

Enable in Vet-AI:

- `VETAI_MLAIR_WORKER_ENABLED=true` (requires `MLAIR_ENABLED=true` and `MLAIR_AUTH_TOKEN` set to a token MLAir accepts for `/v1/tasks/*`, typically the same value as MLAir’s `ML_AIR_WORKER_TOKEN` or a maintainer token).

See `env/.env.example` for `VETAI_MLAIR_WORKER_*` and `VETAI_MLAIR_CT_*` tuning. No changes to the `ml-air` core repo are required for this worker.

---

## Admin authentication

Protected routes expect:

```http
Authorization: Bearer <ADMIN_TOKEN>
```

`ADMIN_TOKEN` must match the server environment (default in code is only for development).

---

## Training modes

1. **In-process** — triggered via API (`/continuous-training/training/trigger`) inside the same FastAPI process (or thread pool), suitable for dev/small workloads.
2. **EKS / Kubernetes Job** — when `ALLOW_EKS_HYBRID_TRAINING` and infrastructure are enabled; see `continuous_training.py` (`run_eks_training`) and `scripts/train_eks.py` for the worker.
3. **Standalone script** — `scripts/train_eks.py` can run inside `docker/Dockerfile.training` with `DATA_SOURCE=postgres` and `DATABASE_URL`.

After successful training, feedback rows are **marked consumed** (ineligible for immediate retrain) rather than deleted, so dashboards based on DB counts stay consistent.

---

## Observability

- **Prometheus** metrics include inference latency, request counts, feedback counters, training job gauges (see `ai_service/metrics.py` and Grafana dashboards in **vet-microservices** / **vet-infra**).

---

## Testing

```bash
pytest
```

---

## License

This project is part of a veterinary clinic microservices codebase. Use and distribution terms should match your organization’s policy; many Spring Petclinic–derived stacks use **Apache License 2.0** — confirm in your repository if you publish a `LICENSE` file.

---

## Contributing

Issues and pull requests: use your team’s GitHub workflow on [DoAnCN-NguyenTaiPhu-TranMinhNhat/vet-ai](https://github.com/DoAnCN-NguyenTaiPhu-TranMinhNhat/vet-ai).
