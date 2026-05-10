import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai_service.app.api.middleware.request_id import request_id_middleware
from ai_service.app.api.routers.health import router as health_router
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

    refresh_fn = None
    try:
        from ai_service.app.domain.services.training_service import get_refresh_training_metrics

        refresh_fn = get_refresh_training_metrics()
    except Exception:
        refresh_fn = None

    stop_event = asyncio.Event()
    bg_tasks: list[asyncio.Task] = []

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
        bg_tasks.append(asyncio.create_task(_refresh_loop()))

    try:
        from ai_service.app.infrastructure.external.mlair_client import (
            run_mlair_registry_bootstrap_with_retries,
            sync_mlair_project_registry_periodic,
        )

        async def _mlair_registry_bootstrap() -> None:
            try:
                out = await asyncio.to_thread(run_mlair_registry_bootstrap_with_retries)
                logger.info(
                    "MLAir project registry bootstrap: status=%s clinics=%s source=%s",
                    out.get("status"),
                    out.get("catalog_clinic_count"),
                    out.get("catalog_source"),
                )
            except Exception as exc:
                logger.warning("MLAir project registry bootstrap failed: %s", exc)

        bg_tasks.append(asyncio.create_task(_mlair_registry_bootstrap()))

        try:
            resync_sec = float(os.getenv("MLAIR_REGISTRY_RESYNC_SECONDS", "120"))
        except ValueError:
            resync_sec = 120.0

        if resync_sec > 0:

            async def _mlair_registry_periodic() -> None:
                while not stop_event.is_set():
                    try:
                        await asyncio.wait_for(stop_event.wait(), timeout=resync_sec)
                    except asyncio.TimeoutError:
                        pass
                    if stop_event.is_set():
                        break
                    try:
                        out = await asyncio.to_thread(sync_mlair_project_registry_periodic)
                        if out.get("status") not in ("skipped",):
                            logger.debug(
                                "MLAir registry periodic: status=%s clinics=%s",
                                out.get("status"),
                                out.get("catalog_clinic_count"),
                            )
                    except Exception as exc:
                        logger.warning("MLAir registry periodic sync failed: %s", exc)

            bg_tasks.append(asyncio.create_task(_mlair_registry_periodic()))
    except Exception as exc:
        logger.warning("MLAir project registry tasks not scheduled: %s", exc)

    yield
    stop_event.set()
    for t in bg_tasks:
        t.cancel()
        try:
            await t
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

if mlops_router:
    app.include_router(mlops_router)

app.include_router(mlops_v2_router)
