import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_service.app.api.middleware.request_id import request_id_middleware
from ai_service.app.api.routers.health import router as health_router
from ai_service.app.api.routers.internal_mlair import router as internal_mlair_router
from ai_service.app.api.routers.mlair import router as mlair_router
from ai_service.app.api.routers.mlair_registry import router as mlair_registry_router
from ai_service.app.api.routers.mlops import router as mlops_router
from ai_service.app.api.routers.mlops_v2 import router as mlops_v2_router
from ai_service.app.api.routers.models import router as models_router
from ai_service.app.api.routers.predict import load_artifacts, router as predict_router
from ai_service.app.api.routers.training import router as training_router
from ai_service.app.core.logging import configure_logging
from ai_service.app.core.metrics import setup_metrics

logger = logging.getLogger(__name__)

configure_logging()

@asynccontextmanager
async def _app_lifespan(app: FastAPI):
    try:
        load_artifacts()
    except Exception as exc:
        logger.warning("Cannot load artifacts at startup: %s", exc)

    # Optional startup sync: push all discoverable model versions to MLAir
    # so model registry is populated even before any new training run completes.
    startup_sync_flag = os.getenv(
        "VETAI_MLAIR_SYNC_MODELS_ON_STARTUP",
        os.getenv("MLAIR_SYNC_MODELS_ON_STARTUP", "true"),
    )
    if startup_sync_flag.lower() == "true":
        try:
            from ai_service.app.infrastructure.external import mlair_client

            if mlair_client.config_summary().get("enabled"):
                sync_result = mlair_client.sync_all_models_to_mlair()
                logger.info("MLAir startup model sync completed: %s", sync_result)
        except Exception as exc:
            logger.warning("MLAir startup model sync skipped/failed: %s", exc)

    refresh_fn = None
    try:
        from ai_service.app.domain.services.training_service import get_refresh_training_metrics

        refresh_fn = get_refresh_training_metrics()
    except Exception:
        refresh_fn = None

    stop_event = asyncio.Event()
    task: asyncio.Task | None = None
    mlair_worker_task: asyncio.Task | None = None

    async def _refresh_loop() -> None:
        while not stop_event.is_set():
            if refresh_fn is not None:
                try:
                    refresh_fn()
                except Exception as exc:
                    logger.warning("Refresh training metrics failed: %s", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                pass

    if refresh_fn is not None:
        try:
            refresh_fn()
        except Exception as exc:
            logger.warning("Cannot refresh training metrics at startup: %s", exc)
        task = asyncio.create_task(_refresh_loop())

    if os.getenv("VETAI_MLAIR_WORKER_ENABLED", "").lower() == "true":
        try:
            from ai_service.app.infrastructure.external import mlair_client

            if mlair_client.config_summary().get("enabled"):
                from ai_service.app.domain.services.mlair_external_worker import run_mlair_worker_loop

                mlair_worker_task = asyncio.create_task(run_mlair_worker_loop(stop_event))
                logger.info("Vet-AI MLAir external worker loop started")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vet-AI MLAir external worker not started: %s", exc)

    yield
    stop_event.set()
    if mlair_worker_task is not None:
        mlair_worker_task.cancel()
        try:
            await mlair_worker_task
        except BaseException:
            pass
    if task is not None:
        task.cancel()
        try:
            await task
        except BaseException:
            pass


app = FastAPI(
    title="Veterinary Diagnosis AI",
    version="2.0.0",
    description="Veterinary AI Diagnosis System with Champion-Challenger MLOps",
    lifespan=_app_lifespan,
)


app.middleware("http")(request_id_middleware)
setup_metrics(app)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(models_router)
app.include_router(training_router)
app.include_router(mlair_router)
app.include_router(mlair_registry_router)
app.include_router(internal_mlair_router)

if mlops_router:
    app.include_router(mlops_router)

app.include_router(mlops_v2_router)

