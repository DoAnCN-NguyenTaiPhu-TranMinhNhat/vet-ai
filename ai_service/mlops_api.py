"""
MLOps API Endpoints for Veterinary AI System
Provides REST endpoints for drift detection, monitoring, and MLOps management
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .mlops_manager import MLOpsManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/mlops", tags=["MLOps"])

# Global MLOps manager instance
mlops_manager = MLOpsManager()

# Pydantic models for API
class PredictionRequest(BaseModel):
    prediction: str = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    actual: Optional[str] = Field(None, description="Actual class (if available)")
    features: Optional[Dict[str, Any]] = Field(None, description="Input features")

class ReferenceDataRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Training data as list of dictionaries")

class DriftCheckRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Current data for drift detection")

class ConfigUpdateRequest(BaseModel):
    drift_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    accuracy_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    min_training_samples: Optional[int] = Field(None, gt=0)
    max_error_rate: Optional[float] = Field(None, ge=0.0, le=1.0)
    check_interval_hours: Optional[int] = Field(None, gt=0)

# API Endpoints

@router.post("/initialize", summary="Initialize MLOps with reference data")
async def initialize_mlops(request: ReferenceDataRequest):
    """
    Initialize MLOps system with training data for drift detection
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Initialize MLOps manager
        result = mlops_manager.initialize_reference_data(df)
        
        if result.get('status') == 'success':
            return {
                "status": "success",
                "message": "MLOps initialized successfully",
                "details": result
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Initialization failed'))
            
    except Exception as e:
        logger.error(f"Error initializing MLOps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prediction", summary="Process prediction through MLOps")
async def process_prediction(request: PredictionRequest):
    """
    Process a prediction through MLOps pipeline for monitoring
    """
    try:
        # Convert to dict for processing
        prediction_data = request.dict()
        
        # Process through MLOps
        result = mlops_manager.process_prediction(prediction_data)
        
        if result.get('status') == 'success':
            return {
                "status": "success",
                "message": "Prediction processed successfully",
                "training_eligible": result.get('training_eligible', False),
                "timestamp": result.get('timestamp')
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))
            
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/drift-check", summary="Check for data drift")
async def check_data_drift(request: DriftCheckRequest):
    """
    Check for data drift in current data
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data provided")
        
        # Check drift
        result = mlops_manager.check_data_drift(df)
        
        if result.get('status') == 'success':
            return {
                "status": "success",
                "message": "Drift check completed",
                "drift_results": result
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Drift check failed'))
            
    except Exception as e:
        logger.error(f"Error checking drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", summary="Get comprehensive MLOps status")
async def get_mlops_status():
    """
    Get comprehensive MLOps system status
    """
    try:
        status = mlops_manager.get_mlops_status()
        
        return {
            "status": "success",
            "mlops_status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting MLOps status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-metrics", summary="Get model performance metrics")
async def get_model_metrics():
    """
    Get current model performance metrics
    """
    try:
        metrics = mlops_manager.model_monitor.get_model_metrics()
        
        return {
            "status": "success",
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", summary="Run comprehensive health check")
async def run_health_check():
    """
    Run comprehensive health check of MLOps system
    """
    try:
        health = mlops_manager.run_health_check()
        
        return {
            "status": "success",
            "health_check": health
        }
        
    except Exception as e:
        logger.error(f"Error running health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retraining-eligibility", summary="Check if model should be retrained")
async def check_retraining_eligibility():
    """
    Check if model should be retrained based on current metrics
    """
    try:
        recommendation = mlops_manager.should_retrain_model()
        
        return {
            "status": "success",
            "retraining_recommendation": recommendation
        }
        
    except Exception as e:
        logger.error(f"Error checking retraining eligibility: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-trend", summary="Get performance trend over time")
async def get_performance_trend(hours: int = 24):
    """
    Get performance trend over specified time period
    """
    try:
        if hours <= 0 or hours > 168:  # Max 7 days
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        trend = mlops_manager.model_monitor.get_performance_trend(hours=hours)
        
        return {
            "status": "success",
            "performance_trend": trend
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts", summary="Get active alerts")
async def get_alerts():
    """
    Get current active alerts from monitoring system
    """
    try:
        health = mlops_manager.model_monitor.check_model_health()
        alerts = health.get('alerts', [])
        
        return {
            "status": "success",
            "active_alerts": alerts,
            "total_alerts": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/config", summary="Update MLOps configuration")
async def update_config(request: ConfigUpdateRequest):
    """
    Update MLOps configuration parameters
    """
    try:
        # Get current config
        current_config = mlops_manager.config.copy()
        
        # Update with provided values
        update_data = request.dict(exclude_unset=True)
        current_config.update(update_data)
        
        # Update manager config
        mlops_manager.config = current_config
        
        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "updated_config": current_config
        }
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/config", summary="Get current MLOps configuration")
async def get_config():
    """
    Get current MLOps configuration
    """
    try:
        return {
            "status": "success",
            "config": mlops_manager.config
        }
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export", summary="Export MLOps data")
async def export_mlops_data(background_tasks: BackgroundTasks):
    """
    Export comprehensive MLOps data for analysis
    """
    try:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mlops_export_{timestamp}.json"
        filepath = f"/tmp/{filename}"
        
        # Export data in background
        def export_data():
            mlops_manager.export_mlops_data(filepath)
        
        background_tasks.add_task(export_data)
        
        return {
            "status": "success",
            "message": "Export started in background",
            "filename": filename,
            "filepath": filepath
        }
        
    except Exception as e:
        logger.error(f"Error exporting MLOps data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/reset", summary="Reset MLOps system")
async def reset_mlops():
    """
    Reset MLOps system (clear all data and reinitialize)
    """
    try:
        # Create new manager instance
        global mlops_manager
        mlops_manager = MLOpsManager()
        
        return {
            "status": "success",
            "message": "MLOps system reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting MLOps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints

@router.get("/info", summary="Get MLOps system information")
async def get_system_info():
    """
    Get general information about the MLOps system
    """
    try:
        info = {
            "system": "Veterinary AI MLOps",
            "version": "1.0.0",
            "components": [
                "Data Drift Detection (Evidently AI)",
                "Model Performance Monitoring",
                "Training Eligibility Checker",
                "Health Monitoring"
            ],
            "initialized": mlops_manager.reference_data is not None,
            "total_predictions": mlops_manager.model_monitor.model_metrics['total_predictions'],
            "last_health_check": mlops_manager.last_health_check.isoformat() if mlops_manager.last_health_check else None
        }
        
        return {
            "status": "success",
            "system_info": info
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize with sample data for testing
@router.post("/initialize-sample", summary="Initialize with sample data")
async def initialize_with_sample_data():
    """
    Initialize MLOps system with sample data for testing
    """
    try:
        import numpy as np
        
        # Generate sample training data
        np.random.seed(42)
        sample_data = []
        
        for i in range(1000):
            sample_data.append({
                'temperature': float(np.random.normal(38.5, 1.0)),
                'weight_kg': float(np.random.normal(25.0, 5.0)),
                'heart_rate': float(np.random.normal(80, 10)),
                'age_months': int(np.random.randint(1, 180)),
                'animal_type': np.random.choice(['Dog', 'Cat', 'Bird']),
                'gender': np.random.choice(['Male', 'Female']),
                'vaccination_status': np.random.choice(['Yes', 'No']),
                'target_diagnosis': np.random.choice(['Healthy', 'Flu', 'Digestive'])
            })
        
        # Initialize with sample data
        df = pd.DataFrame(sample_data)
        result = mlops_manager.initialize_reference_data(df)
        
        if result.get('status') == 'success':
            return {
                "status": "success",
                "message": "MLOps initialized with sample data",
                "details": result
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Initialization failed'))
            
    except Exception as e:
        logger.error(f"Error initializing with sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting MLOps API Server...")
    print("Available endpoints:")
    print("  POST /mlops/initialize - Initialize with reference data")
    print("  POST /mlops/prediction - Process prediction")
    print("  POST /mlops/drift-check - Check data drift")
    print("  GET  /mlops/status - Get MLOps status")
    print("  GET  /mlops/health - Run health check")
    print("  GET  /mlops/model-metrics - Get model metrics")
    print("  GET  /mlops/retraining-eligibility - Check retraining eligibility")
    print("  POST /mlops/initialize-sample - Initialize with sample data")
    print("  GET  /mlops/info - Get system info")
    print("\n📊 MLOps API ready for monitoring!")
    
    # Run server (for testing)
    uvicorn.run(router, host="0.0.0.0", port=8001, log_level="info")
