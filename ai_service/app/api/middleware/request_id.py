import contextvars
import logging
import time
import uuid

from fastapi import Request, Response

request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class RequestIdLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_ctx.get()
        setattr(record, "request_id", rid or "-")
        return True


async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    token = request_id_ctx.set(rid)
    start = time.time()
    try:
        response: Response = await call_next(request)
    finally:
        request_id_ctx.reset(token)
    response.headers["X-Request-Id"] = rid
    logging.getLogger(__name__).info(
        "request completed",
        extra={"path": request.url.path, "method": request.method, "ms": int((time.time() - start) * 1000)},
    )
    return response
