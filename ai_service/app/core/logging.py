import logging

from ai_service.app.api.middleware.request_id import RequestIdLogFilter
from ai_service.app.core.settings import settings


def configure_logging() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    logging.getLogger().addFilter(RequestIdLogFilter())
