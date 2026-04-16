from typing import Any

from ai_service.app.api.routers import predict as predict_router


def load_artifacts() -> None:
    predict_router.load_artifacts()


def set_active_model_and_reload(model_version: str) -> None:
    predict_router.set_active_model_and_reload(model_version)


def clear_artifact_cache() -> None:
    predict_router.clear_artifact_cache()


def model_info() -> dict[str, Any]:
    return predict_router.model_info()
