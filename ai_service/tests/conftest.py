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
    model_dir = root / "models" / "v2"
    os.environ.setdefault("MODEL_DIR", str(model_dir))


@pytest.fixture()
def client() -> TestClient:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from main import app

    with TestClient(app) as c:
        yield c
