import os
import pathlib
import sys

import pytest

from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def _configure_model_dir() -> None:
    # Ensure tests can load artifacts regardless of where pytest is invoked.
    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    
    # Check for models in ai_service directory first
    ai_service_models = root / "ai_service" / "models" / "v2"
    if ai_service_models.exists():
        model_dir = ai_service_models
    else:
        model_dir = root / "models" / "v2"
    
    os.environ.setdefault("MODEL_DIR", str(model_dir))
    
    # Add ai_service to path for relative imports
    ai_service_path = root / "ai_service"
    if ai_service_path.exists():
        sys.path.insert(0, str(ai_service_path))


@pytest.fixture()
def client() -> TestClient:
    import sys
    import os
    # Add parent directory to path for imports
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, parent_dir)
    
    # Add ai_service to path for relative imports
    ai_service_dir = os.path.join(parent_dir, "ai_service")
    if os.path.exists(ai_service_dir):
        sys.path.insert(0, ai_service_dir)
    
    # Import main with proper path setup
    try:
        from main import app
    except ImportError:
        # Fallback for CI/CD environment
        import sys
        sys.path.insert(0, os.path.join(parent_dir, "ai_service"))
        from main import app

    with TestClient(app) as c:
        yield c
