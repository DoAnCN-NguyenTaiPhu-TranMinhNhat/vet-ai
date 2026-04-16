from fastapi import Depends

from ai_service.app.core.auth import verify_admin as _verify_admin


async def require_admin(ok: bool = Depends(_verify_admin)) -> bool:
    return ok
