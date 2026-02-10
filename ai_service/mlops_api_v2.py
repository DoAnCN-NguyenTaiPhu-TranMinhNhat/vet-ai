"""
MLOps API v2 - Champion-Challenger Workflow Endpoints
Handles model promotion, approval, and rollback operations
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import logging
from datetime import datetime

from .mlops_champion_challenger import ChampionChallengerManager, ModelStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mlops/v2", tags=["mlops-champion-challenger"])
security = HTTPBearer()

# Pydantic models
class StagingModelRequest(BaseModel):
    model_path: str = Field(..., description="Path to trained model")
    model_version: str = Field(..., description="Model version identifier")
    training_metrics: Dict = Field(..., description="Training metrics")

class ApprovalRequest(BaseModel):
    approver: str = Field(..., description="Approver name/ID")
    comments: Optional[str] = Field(None, description="Approval comments")

class RollbackRequest(BaseModel):
    archived_model_version: str = Field(..., description="Model version to rollback to")
    reason: Optional[str] = Field("Manual rollback", description="Rollback reason")

# Dependency for admin authentication
async def verify_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify admin credentials for approval operations"""
    # In production, implement proper JWT validation
    token = credentials.credentials
    if token != os.getenv("ADMIN_TOKEN", "admin-secret"):
        raise HTTPException(status_code=403, detail="Admin access required")
    return credentials

# Initialize manager
challenger_manager = ChampionChallengerManager()

@router.post("/staging/register")
async def register_staging_model(request: StagingModelRequest):
    """Register new trained model to STAGING environment"""
    try:
        result = challenger_manager.register_staging_model(
            model_path=request.model_path,
            training_metrics=request.training_metrics,
            model_version=request.model_version
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Failed to register staging model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/staging/{model_version}/evaluate")
async def evaluate_staging_model(model_version: str):
    """Evaluate staging model against production"""
    try:
        result = challenger_manager.evaluate_staging_model(model_version)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/staging/{model_version}/request-approval")
async def request_approval(model_version: str, requestor: str = "system"):
    """Create approval request for model promotion"""
    try:
        result = challenger_manager.request_approval(model_version, requestor)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Approval request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/staging/{model_version}/approve")
async def approve_model_promotion(
    model_version: str, 
    request: ApprovalRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_admin)
):
    """Approve and promote staging model to production (Admin only)"""
    try:
        result = challenger_manager.approve_model_promotion(
            staging_model_version=model_version,
            approver=request.approver,
            comments=request.comments
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Model approval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/staging/{model_version}/reject")
async def reject_model_promotion(
    model_version: str,
    request: ApprovalRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_admin)
):
    """Reject staging model promotion (Admin only)"""
    try:
        result = challenger_manager.reject_model_promotion(
            staging_model_version=model_version,
            approver=request.approver,
            reason=request.comments or "Rejected by admin"
        )
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Model rejection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rollback")
async def rollback_to_archived(
    request: RollbackRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_admin)
):
    """Rollback to archived model version (Admin only)"""
    try:
        result = challenger_manager.rollback_to_archived(request.archived_model_version)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/registry/status")
async def get_registry_status():
    """Get current status of all models in registry"""
    try:
        result = challenger_manager.get_model_registry_status()
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Failed to get registry status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/staging/{model_version}/details")
async def get_staging_model_details(model_version: str):
    """Get detailed information about staging model"""
    try:
        import os
        import json
        
        staging_path = f"/app/models/staging/{model_version}"
        details = {"model_version": model_version}
        
        # Load metadata
        metadata_path = f"{staging_path}/staging_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                details["metadata"] = json.load(f)
        
        # Load evaluation
        eval_path = f"{staging_path}/evaluation.json"
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                details["evaluation"] = json.load(f)
        
        # Load approval
        approval_path = f"{staging_path}/approval.json"
        if os.path.exists(approval_path):
            with open(approval_path, "r") as f:
                details["approval"] = json.load(f)
        
        return {"status": "success", "data": details}
    except Exception as e:
        logger.error(f"Failed to get model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/production/current")
async def get_current_production_model():
    """Get current production model details"""
    try:
        import os
        import json
        
        current_prod = challenger_manager._get_current_production_model()
        if not current_prod:
            return {"status": "success", "data": {"message": "No production model found"}}
        
        model_version = os.path.basename(current_prod)
        metadata_path = f"{current_prod}/production_metadata.json"
        
        details = {"model_version": model_version, "path": current_prod}
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                details["metadata"] = json.load(f)
        
        return {"status": "success", "data": details}
    except Exception as e:
        logger.error(f"Failed to get production model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for MLOps service"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
