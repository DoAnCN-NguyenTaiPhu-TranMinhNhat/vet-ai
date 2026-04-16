from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "UP"}


@router.get("/readyz", include_in_schema=False)
def readyz() -> dict[str, str]:
    return {"status": "UP"}


@router.get("/livez", include_in_schema=False)
def livez() -> dict[str, str]:
    return {"status": "UP"}
