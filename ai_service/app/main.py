import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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
    task: asyncio.Task | None = None

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
    yield
    stop_event.set()
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

_UI_DIR = Path(__file__).resolve().parents[1] / "ui"
_UI_STATIC_DIR = _UI_DIR / "static"
if _UI_STATIC_DIR.exists():
    app.mount("/mlops-ui/static", StaticFiles(directory=str(_UI_STATIC_DIR)), name="mlops-ui-static")


@app.get("/mlops-ui", include_in_schema=False)
def mlops_ui() -> HTMLResponse:
    index = _UI_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h3>UI not found</h3>", status_code=404)
    return HTMLResponse(index.read_text(encoding="utf-8"))


app.middleware("http")(request_id_middleware)
setup_metrics(app)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(models_router)
app.include_router(training_router)

if mlops_router:
    app.include_router(mlops_router)

app.include_router(mlops_v2_router)

