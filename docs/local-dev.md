# Local Development

## Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `env/.env.example` to your own env file and export variables.

## Run API

```bash
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
