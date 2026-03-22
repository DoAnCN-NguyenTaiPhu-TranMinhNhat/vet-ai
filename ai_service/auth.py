import os

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)


async def verify_admin(credentials: HTTPAuthorizationCredentials | None = Depends(security)):
    """
    Simple admin auth: Authorization: Bearer <ADMIN_TOKEN>
    """
    expected = os.getenv("ADMIN_TOKEN", "admin-secret")
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing admin token")
    if credentials.credentials != expected:
        raise HTTPException(status_code=403, detail="Admin access required")
    return True

