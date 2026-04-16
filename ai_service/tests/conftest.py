import os
import pathlib
import sys
import tempfile

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
    # Avoid PermissionError for default registry path on dev/CI runners.
    os.environ.setdefault(
        "VETAI_CHAMPION_REGISTRY_PATH",
        os.path.join(tempfile.gettempdir(), "vetai-champion-registry"),
    )


@pytest.fixture()
def client() -> TestClient:
    # Import app from new package layout
    from ai_service.app.main import app

    with TestClient(app) as c:
        yield c
