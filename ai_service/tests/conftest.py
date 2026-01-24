import os
import pathlib
import sys

import pytest

from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _configure_model_dir() -> None:
    # Ensure tests can load artifacts regardless of where pytest is invoked.
    root = pathlib.Path(__file__).resolve().parents[2]  # Go up to vet-ai root
    sys.path.insert(0, str(root))
    
    # Check for models in ai_service directory first
    ai_service_models = root / "ai_service" / "models" / "v2"
    if ai_service_models.exists():
        model_dir = ai_service_models
    else:
        model_dir = root / "models" / "v2"
    
    os.environ.setdefault("MODEL_DIR", str(model_dir))


@pytest.fixture()
def client() -> TestClient:
    # Import app using absolute package path
    from ai_service.main import app

    with TestClient(app) as c:
        yield c
