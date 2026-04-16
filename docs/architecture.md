# Architecture

## High-level

- `ai_service/` contains the FastAPI application, MLOps modules, and training flow.
- `docker/` stores compose files for local and production-like environments.
- `scripts/` stores helper scripts for local and EKS training workflows.
- `monitoring/` stores Prometheus/Grafana related assets.

## Application boundaries

- `ai_service/app/api/routers`: HTTP layer only.
- `ai_service/app/domain`: request schemas and business services.
- `ai_service/app/infrastructure`: external clients and storage adapters.
- `ai_service/app/main.py`: app composition and middleware wiring.
