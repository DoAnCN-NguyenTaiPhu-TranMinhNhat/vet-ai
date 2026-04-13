# Vet-AI

FastAPI service for **veterinary diagnosis** (sklearn RandomForest), **continuous learning** from clinician feedback, **MLOps** (MLflow, optional S3 artifacts), **Prometheus** metrics, and a small **MLOps UI**.

---

## Overview

Vet-AI exposes a REST API consumed by the Spring **genai-service** (and similar clients). It loads trained models from disk, runs inference on structured visit features, logs predictions and feedback to **PostgreSQL** for retraining, and can promote models per clinic (multi-tenant) or globally.

| Topic | Description |
|--------|-------------|
| Inference | `POST /predict` — symptoms and visit features → ranked diagnoses |
| Feedback | `POST /continuous-training/feedback` — accept/reject + final diagnosis for training |
| Training | In-process or **EKS/K8s** job (`train_eks.py`) — see `Dockerfile.training` |
| Registry | Active model pins, MLflow runs, optional champion/challenger under `/mlops/v2` |

OpenAPI docs: **`GET /docs`** (Swagger UI) when the app is running.

---

## Features

- **Inference** with per-request latency and confidence; optional **clinic-scoped** active model.
- **Continuous training**: policy thresholds, job history, CSV bootstrap (admin), feedback pool (GLOBAL vs CLINIC_ONLY).
- **Postgres** persistence for predictions, feedback, and training jobs (`DATABASE_URL`).
- **MLflow** experiment tracking (`MLFLOW_TRACKING_URI`).
- **Prometheus**: `/metrics` (via `prometheus-fastapi-instrumentator`) plus custom counters/histograms in `ai_service/metrics.py`.
- **MLOps UI** (static): **`GET /mlops-ui`** — training triggers, policy, registry status (admin token required for mutating calls from the browser).

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

- **Python 3.11** (see `Dockerfile`)
- **PostgreSQL** for continuous training / feedback (same DB can be shared with MLflow in dev; production often uses separate DB/schema)
- **MLflow** server (optional but recommended for experiment tracking)
- **Docker** or **Podman** (optional) — images: `Dockerfile`, `Dockerfile.training`

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
| **`.env.exemple`** | **Committed template.** Full variable set and defaults. | No — template only |
| **`.env.local`** | **Local runtime file** (gitignored). Use this for local `uvicorn` / scripts. | No — load with `source` or inject in your shell |

In Docker, **`vet-microservices`** Compose usually injects variables via **`../vet-infra/.env`** for the `vet-ai` service — not from these files unless you wire them yourself.

Start from **`.env.exemple`** in this repo. Copy values into **`.env.local`** for local runs, or merge needed keys into **`vet-infra/.env`** when using `vet-microservices` Compose.

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

**API image** (`Dockerfile`):

```bash
docker build -t vet-ai:latest .
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e ADMIN_TOKEN=... \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  vet-ai:latest
```

**Training worker image** (`Dockerfile.training`) — used for CPU-heavy jobs; entrypoint scripts include `train_eks.py` / local training helpers.

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
| `/metrics` | Prometheus scrape endpoint. |
| `/mlops-ui` | Static MLOps UI. |

Full schemas: **`/docs`**.

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
2. **EKS / Kubernetes Job** — when `ALLOW_EKS_HYBRID_TRAINING` and infrastructure are enabled; see `continuous_training.py` (`run_eks_training`) and `train_eks.py` for the worker.
3. **Standalone script** — `train_eks.py` can run inside `Dockerfile.training` with `DATA_SOURCE=postgres` and `DATABASE_URL`.

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
