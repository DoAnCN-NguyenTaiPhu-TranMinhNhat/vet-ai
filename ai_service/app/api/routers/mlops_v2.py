from fastapi import APIRouter

from ai_service.app.mlops.champion_challenger import ChampionChallengerManager


router = APIRouter(prefix="/mlops/v2", tags=["mlops-champion-challenger"])
_cc_manager = ChampionChallengerManager()


@router.get("/registry/status", summary="Champion-challenger registry folders status")
async def get_registry_status():
    return {
        "status": "success",
        "registry": _cc_manager.get_model_registry_status(),
    }
