# Local Development

## Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. From the repo root: `cp .env.example .env` and edit `.env`. Load variables before running, for example:
   `set -a && source .env && set +a`

## Run API

```bash
cd /path/to/vet-ai
set -a && source .env && set +a
uvicorn ai_service.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Run tests

```bash
pytest
```

## Code quality

```bash
pre-commit run -a
```
