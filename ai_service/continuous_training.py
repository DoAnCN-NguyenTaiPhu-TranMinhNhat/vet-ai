"""
Continuous Training endpoints for FastAPI service
Handles prediction logging and training trigger logic
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import HTTPException, BackgroundTasks, APIRouter
import asyncio
import logging
import os
import json
import uuid
import time
import subprocess
import boto3

# Import kubernetes only when needed for EKS training
try:
    from kubernetes import client, config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    client = None
    config = None

# Import training engine
try:
    from ai_service.training_engine import execute_training
except ImportError:
    from training_engine import execute_training

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/continuous-training", tags=["continuous-training"])

# Configuration
TRAINING_THRESHOLD = 10  # Reduced from 100 for easier testing
TRAINING_WINDOW_DAYS = int(os.getenv("TRAINING_WINDOW_DAYS", "30"))
EKS_CLUSTER_NAME = os.getenv("EKS_CLUSTER_NAME", "vet-ai-dev")
EKS_REGION = os.getenv("EKS_REGION", "us-east-1")
TRAINING_NODE_GROUP = os.getenv("TRAINING_NODE_GROUP", "training-nodes")

# Pydantic models
class PredictionLog(BaseModel):
    visit_id: int
    pet_id: int
    prediction_input: Dict
    prediction_output: Dict
    model_version: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    top_k_predictions: List[Dict]
    veterinarian_id: Optional[int] = None
    clinic_id: Optional[int] = None

class DoctorFeedback(BaseModel):
    prediction_id: int
    final_diagnosis: str
    is_correct: bool
    confidence_rating: int = Field(ge=1, le=5)
    comments: Optional[str] = None
    veterinarian_id: int
    is_training_eligible: bool = True
    data_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)

class TrainingTriggerRequest(BaseModel):
    trigger_type: str = Field(pattern="^(scheduled|manual|automatic)$")
    trigger_reason: Optional[str] = None
    force: bool = False  # Override threshold check
    training_mode: str = Field(default="local", pattern="^(local|eks_hybrid)$")
    eks_node_group: Optional[str] = None

class TrainingStatus(BaseModel):
    training_id: int
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_predictions: int
    eligible_feedback_count: int
    previous_model_version: Optional[str]
    new_model_version: Optional[str]
    training_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    f1_score: Optional[float]
    is_deployed: bool
    error_message: Optional[str]

# Database functions (currently in-memory, ready for database migration)
async def log_prediction(prediction: PredictionLog) -> int:
    """Log prediction to storage"""
    logger.info(f"Logging prediction for visit {prediction.visit_id}")
    
    prediction_data = {
        "id": hash(f"{prediction.visit_id}-{datetime.now()}"),
        "visit_id": prediction.visit_id,
        "pet_id": prediction.pet_id,
        "prediction_input": prediction.prediction_input,
        "prediction_output": prediction.prediction_output,
        "model_version": prediction.model_version,
        "confidence_score": prediction.confidence_score,
        "top_k_predictions": prediction.top_k_predictions,
        "veterinarian_id": prediction.veterinarian_id,
        "clinic_id": prediction.clinic_id,
        "timestamp": datetime.now()
    }
    prediction_logs.append(prediction_data)
    
    logger.info(f"Prediction logged successfully. Total predictions: {len(prediction_logs)}")
    return prediction_data["id"]

async def save_feedback(feedback: DoctorFeedback) -> bool:
    """Save doctor feedback to storage"""
    logger.info(f"Saving feedback for prediction {feedback.prediction_id}")
    
    # Store in memory (ready for database migration)
    feedback_data_item = {
        "prediction_id": feedback.prediction_id,
        "final_diagnosis": feedback.final_diagnosis,
        "is_correct": feedback.is_correct,
        "confidence_rating": feedback.confidence_rating,
        "comments": feedback.comments,
        "veterinarian_id": feedback.veterinarian_id,
        "is_training_eligible": feedback.is_training_eligible,
        "data_quality_score": feedback.data_quality_score,
        "timestamp": datetime.now()
    }
    feedback_data.append(feedback_data_item)
    
    logger.info(f"Feedback saved. Total feedback count: {len(feedback_data)}")
    return True

# Production-ready in-memory storage
# Ready for database migration to PostgreSQL/Redis
prediction_logs: List[PredictionLog] = []
feedback_data: List[DoctorFeedback] = []
training_jobs: Dict[str, TrainingStatus] = {}

# Initialize clean storage
prediction_logs.clear()
feedback_data.clear()
training_jobs.clear()

training_counter = 0  # Sequential counter for reliable IDs

logger.info("Production-ready continuous training system initialized")

async def count_eligible_feedback(days: int = TRAINING_WINDOW_DAYS) -> int:
    """Count eligible feedback in the last N days"""
    # Count actual feedback from in-memory storage
    cutoff_date = datetime.now() - timedelta(days=days)
    eligible_count = 0
    
    for feedback in feedback_data:
        # Check if feedback is within window and eligible
        feedback_date = feedback.get('timestamp', datetime.now())
        if feedback_date >= cutoff_date and feedback.get('is_training_eligible', True):
            eligible_count += 1
    
    logger.info(f"Current eligible feedback count: {eligible_count}")
    return eligible_count

async def trigger_training_job(request: TrainingTriggerRequest) -> int:
    """Trigger a training job and return training ID"""
    logger.info(f"Triggering {request.training_mode} training: {request.trigger_type}")
    
    # Generate reliable sequential training ID
    global training_counter
    training_counter += 1
    training_id = training_counter
    
    # Store training job
    training_jobs[training_id] = {
        "training_id": training_id,
        "status": "running",
        "start_time": datetime.now(),
        "end_time": None,
        "total_predictions": len(prediction_logs),
        "eligible_feedback_count": len(feedback_data),
        "previous_model_version": "v2.0",
        "new_model_version": None,
        "training_accuracy": None,
        "validation_accuracy": None,
        "f1_score": None,
        "is_deployed": False,
        "error_message": None,
        "trigger_type": request.trigger_type,
        "training_mode": request.training_mode
    }
    
    # Start actual training
    logger.info(f"Starting actual training for job {training_id}")
    task = asyncio.create_task(execute_actual_training(training_id, request.training_mode))
    logger.info(f"Training task created: {task}")
    
    return training_id

async def execute_actual_training(training_id: int, training_mode: str):
    """Execute actual ML training"""
    global feedback_data
    
    try:
        logger.info(f"Starting actual {training_mode} training for job {training_id}")
        
        if training_mode == "eks_hybrid":
            # Scale up node group and run training on EKS
            result = await run_eks_training(training_id)
        else:
            # Run local training
            result = execute_training(feedback_data, prediction_logs, "local")
        
        if training_id in training_jobs:
            if result['status'] == 'completed':
                # Update training job with completion
                training_jobs[training_id].update({
                    "status": "completed",
                    "end_time": datetime.now(),
                    "new_model_version": result.get('model_version', 'v2.1'),
                    "training_accuracy": result['training_metrics'].get('training_accuracy'),
                    "validation_accuracy": result['training_metrics'].get('validation_accuracy'),
                    "f1_score": result['training_metrics'].get('validation_f1'),
                    "is_deployed": True
                })
                
                # Reset feedback count after successful training
                feedback_data.clear()
                logger.info(f"Feedback data reset after training completion. New count: {len(feedback_data)}")
                
                logger.info(f"Training {training_id} completed successfully!")
            else:
                # Training failed
                training_jobs[training_id].update({
                    "status": "failed",
                    "error_message": result.get('error', 'Unknown error')
                })
        else:
            logger.error(f"Training job {training_id} not found in storage")
            
    except Exception as e:
        logger.error(f"Training execution failed for {training_id}: {e}")
        if training_id in training_jobs:
            training_jobs[training_id].update({
                "status": "failed",
                "error_message": str(e)
            })

async def run_eks_training(training_id: int) -> Dict[str, Any]:
    """Run training on EKS with node group scaling"""
    try:
        logger.info(f"Scaling up EKS training node group")
        
        # Initialize boto3 and kubernetes clients
        eks_client = boto3.client('eks', region_name=EKS_REGION)
        
        # Scale up node group
        response = eks_client.update_nodegroup_config(
            clusterName=EKS_CLUSTER_NAME,
            nodegroupName=TRAINING_NODE_GROUP,
            scalingConfig={
                'minSize': 1,
                'desiredSize': 1,
                'maxSize': 1
            }
        )
        
        logger.info(f"Node group scaling initiated: {response['update']['status']}")
        
        # Wait for node group to be ready
        await wait_for_node_group_ready(eks_client, TRAINING_NODE_GROUP)
        
        # Run training job on EKS
        training_result = await run_training_job_on_eks(training_id)
        
        # Scale down node group
        logger.info(f"Scaling down EKS training node group")
        eks_client.update_nodegroup_config(
            clusterName=EKS_CLUSTER_NAME,
            nodegroupName=TRAINING_NODE_GROUP,
            scalingConfig={
                'minSize': 0,
                'desiredSize': 0,
                'maxSize': 1
            }
        )
        
        return training_result
        
    except Exception as e:
        logger.error(f"EKS training failed: {e}")
        return {'status': 'failed', 'error': str(e)}

async def wait_for_node_group_ready(eks_client, nodegroup_name: str, timeout: int = 600):
    """Wait for node group to become ready"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = eks_client.describe_nodegroup(
            clusterName=EKS_CLUSTER_NAME,
            nodegroupName=nodegroup_name
        )
        
        status = response['nodegroup']['status']
        if status == 'ACTIVE':
            logger.info(f"Node group {nodegroup_name} is ready")
            return True
        
        logger.info(f"Node group status: {status}, waiting...")
        await asyncio.sleep(30)
    
    raise TimeoutError(f"Node group {nodegroup_name} did not become ready in {timeout} seconds")

async def run_training_job_on_eks(training_id: int) -> Dict[str, Any]:
    """Run training job as Kubernetes pod on EKS"""
    if not KUBERNETES_AVAILABLE:
        return {'status': 'failed', 'error': 'Kubernetes client not available. Install kubernetes package.'}
    
    try:
        # Load kubernetes config
        config.load_kube_config()
        k8s_client = client.CoreV1Api()
        
        # Create training job manifest
        job_manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": f"vet-ai-training-{training_id}",
                "namespace": "default"
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "training",
                            "image": "vet-ai/training:latest",
                            "env": [
                                {"name": "TRAINING_ID", "value": str(training_id)},
                                {"name": "MLFLOW_TRACKING_URI", "value": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")}
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "1000m",
                                    "memory": "4Gi"
                                },
                                "limits": {
                                    "cpu": "2000m",
                                    "memory": "8Gi"
                                }
                            }
                        }],
                        "restartPolicy": "OnFailure",
                        "nodeSelector": {
                            "node-group": "training"
                        }
                    }
                }
            }
        }
        
        # Submit job
        batch_client = client.BatchV1Api()
        job = batch_client.create_namespaced_job(namespace="default", body=job_manifest)
        
        logger.info(f"Training job submitted: {job.metadata.name}")
        
        # Wait for job completion
        await wait_for_job_completion(batch_client, job.metadata.name, "default")
        
        # Get job results
        job_status = batch_client.read_namespaced_job_status(job.metadata.name, "default")
        
        if job_status.status.succeeded:
            # Get training results from pod logs
            pod_name = await get_training_pod_name(k8s_client, job.metadata.name)
            logs = k8s_client.read_namespaced_pod_log(pod_name, "default")
            
            # Parse results from logs
            result = parse_training_results(logs)
            return result
        else:
            return {'status': 'failed', 'error': 'Training job failed on EKS'}
            
    except Exception as e:
        logger.error(f"Failed to run training job on EKS: {e}")
        return {'status': 'failed', 'error': str(e)}

async def wait_for_job_completion(batch_client, job_name: str, namespace: str, timeout: int = 3600):
    """Wait for Kubernetes job to complete"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        job = batch_client.read_namespaced_job_status(job_name, namespace)
        
        if job.status.succeeded or job.status.failed:
            logger.info(f"Job {job_name} completed with status: {job.status}")
            return
        
        logger.info(f"Job {job_name} still running...")
        await asyncio.sleep(60)
    
    raise TimeoutError(f"Job {job_name} did not complete in {timeout} seconds")

async def get_training_pod_name(k8s_client, job_name: str) -> str:
    """Get the pod name for a training job"""
    pods = k8s_client.list_namespaced_pod(
        namespace="default",
        label_selector=f"job-name={job_name}"
    )
    
    if pods.items:
        return pods.items[0].metadata.name
    else:
        raise ValueError(f"No pod found for job {job_name}")

def parse_training_results(logs: str) -> Dict[str, Any]:
    """Parse training results from pod logs"""
    try:
        # Look for JSON result in logs
        for line in logs.split('\n'):
            if line.startswith('TRAINING_RESULT:'):
                result_json = line.replace('TRAINING_RESULT:', '')
                return json.loads(result_json)
        
        return {'status': 'failed', 'error': 'No training result found in logs'}
    except Exception as e:
        return {'status': 'failed', 'error': f'Failed to parse results: {e}'}

async def get_training_status(training_id: int) -> Optional[TrainingStatus]:
    """Get training status by ID"""
    # Check if training job exists
    if training_id in training_jobs:
        job_data = training_jobs[training_id]
        return TrainingStatus(**job_data)
    
    # Return default pending status for unknown jobs
    return TrainingStatus(
        training_id=training_id,
        status="pending",
        start_time=None,
        end_time=None,
        total_predictions=len(prediction_logs),
        eligible_feedback_count=len(feedback_data),
        previous_model_version="v2.0",
        new_model_version=None,
        training_accuracy=None,
        validation_accuracy=None,
        f1_score=None,
        is_deployed=False,
        error_message="Training job not found"
    )

# API Endpoints
@router.post("/predictions/log", status_code=201)
async def log_prediction_endpoint(prediction: PredictionLog):
    """Log a prediction for continuous training"""
    try:
        prediction_id = await log_prediction(prediction)
        return {"prediction_id": prediction_id, "status": "logged"}
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
        raise HTTPException(status_code=500, detail="Failed to log prediction")

@router.post("/feedback", status_code=201)
async def save_feedback_endpoint(feedback: DoctorFeedback):
    """Save doctor feedback for a prediction"""
    try:
        success = await save_feedback(feedback)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to save feedback")
        
        # Check if we should trigger retraining
        eligible_count = await count_eligible_feedback()
        should_trigger = eligible_count >= TRAINING_THRESHOLD
        
        return {
            "status": "saved",
            "eligible_feedback_count": eligible_count,
            "training_threshold": TRAINING_THRESHOLD,
            "should_trigger_training": should_trigger
        }
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@router.get("/training/status")
async def get_training_status_endpoint(training_id: int):
    """Get training job status"""
    try:
        status = await get_training_status(training_id)
        if not status:
            raise HTTPException(status_code=404, detail="Training job not found")
        return status
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training status")

@router.post("/training/trigger", status_code=201)
async def trigger_training_endpoint(
    request: TrainingTriggerRequest,
    background_tasks: BackgroundTasks
):
    """Trigger a training job"""
    try:
        # Check threshold unless forced
        if not request.force:
            eligible_count = await count_eligible_feedback()
            if eligible_count < TRAINING_THRESHOLD:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Insufficient data: {eligible_count} < {TRAINING_THRESHOLD}"
                )
        
        training_id = await trigger_training_job(request)
        
        # Add background task to monitor training
        background_tasks.add_task(
            monitor_training_progress,
            training_id
        )
        
        return {
            "training_id": training_id,
            "status": "triggered",
            "trigger_type": request.trigger_type
        }
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger training")

@router.get("/training/eligibility")
async def check_training_eligibility():
    """Check if system is eligible for retraining"""
    try:
        eligible_count = await count_eligible_feedback()
        is_eligible = eligible_count >= TRAINING_THRESHOLD
        
        return {
            "eligible_feedback_count": eligible_count,
            "training_threshold": TRAINING_THRESHOLD,
            "is_eligible_for_training": is_eligible,
            "training_window_days": TRAINING_WINDOW_DAYS,
            "next_check_date": datetime.now() + timedelta(days=1)
        }
    except Exception as e:
        logger.error(f"Failed to check eligibility: {e}")
        raise HTTPException(status_code=500, detail="Failed to check eligibility")

@router.get("/training/history")
async def get_training_history(limit: int = 10, offset: int = 0):
    """Get training job history from actual training jobs"""
    try:
        # Get all training jobs and sort by ID (most recent first)
        all_jobs = list(training_jobs.values())
        all_jobs.sort(key=lambda x: x.get("training_id", 0), reverse=True)
        
        # Apply pagination
        total_count = len(all_jobs)
        paginated_jobs = all_jobs[offset:offset + limit]
        
        # Format response
        training_runs = []
        for job in paginated_jobs:
            training_runs.append({
                "training_id": job.get("training_id"),
                "status": job.get("status"),
                "trigger_type": job.get("trigger_type"),
                "created_at": job.get("start_time"),
                "training_accuracy": job.get("training_accuracy"),
                "validation_accuracy": job.get("validation_accuracy"),
                "f1_score": job.get("f1_score"),
                "is_deployed": job.get("is_deployed", False),
                "error_message": job.get("error_message"),
                "training_mode": job.get("training_mode"),
                "previous_model_version": job.get("previous_model_version"),
                "new_model_version": job.get("new_model_version")
            })
        
        return {
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "training_runs": training_runs
        }
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training history")

async def get_training_overview():
    """Get training system overview"""
    try:
        # Count active training jobs
        active_jobs = sum(1 for job in training_jobs.values() if job.get('status') == 'running')
        completed_jobs = sum(1 for job in training_jobs.values() if job.get('status') == 'completed')
        failed_jobs = sum(1 for job in training_jobs.values() if job.get('status') == 'failed')
        
        # Get recent training jobs
        recent_jobs = sorted(
            training_jobs.values(), 
            key=lambda x: x.get('start_time', datetime.min), 
            reverse=True
        )[:5]
        
        return {
            "system_status": "active",
            "current_feedback_count": len(feedback_data),
            "training_threshold": TRAINING_THRESHOLD,
            "is_eligible_for_training": len(feedback_data) >= TRAINING_THRESHOLD,
            "training_jobs": {
                "active": active_jobs,
                "completed": completed_jobs,
                "failed": failed_jobs,
                "total": len(training_jobs)
            },
            "recent_training_jobs": recent_jobs,
            "data_collection": {
                "total_predictions": len(prediction_logs),
                "total_feedback": len(feedback_data),
                "eligible_feedback": len([f for f in feedback_data if f.get('is_training_eligible', True)])
            }
        }
    except Exception as e:
        logger.error(f"Failed to get training overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get training overview")

@router.post("/reset-all")
async def reset_all_data():
    """Reset all predictions and feedback data for clean testing"""
    try:
        prediction_count = len(prediction_logs)
        feedback_count = len(feedback_data)
        
        prediction_logs.clear()
        feedback_data.clear()
        
        # Reset prediction counter in main module
        import sys
        if 'ai_service.main' in sys.modules:
            sys.modules['ai_service.main']._prediction_counter = 0
        
        logger.info(f"Reset all data: {prediction_count} predictions, {feedback_count} feedback entries")
        
        return {
            "status": "success",
            "message": f"Reset {prediction_count} predictions and {feedback_count} feedback entries",
            "current_predictions": 0,
            "current_feedback": 0
        }
    except Exception as e:
        logger.error(f"Failed to reset all data: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset all data")

@router.get("/predictions/all")
async def get_all_predictions():
    """Get all prediction logs for debugging"""
    try:
        return {
            "status": "success",
            "total_predictions": len(prediction_logs),
            "predictions": prediction_logs
        }
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get predictions")

@router.get("/predictions/count")
async def get_predictions_count():
    """Get total number of predictions"""
    try:
        return {
            "status": "success",
            "total_predictions": len(prediction_logs)
        }
    except Exception as e:
        logger.error(f"Failed to get prediction count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get prediction count")

@router.get("/feedback/count")
async def get_feedback_count():
    """Get total number of feedback entries"""
    try:
        return {
            "status": "success",
            "total_feedback": len(feedback_data)
        }
    except Exception as e:
        logger.error(f"Failed to get feedback count: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feedback count")

@router.post("/feedback/reset")
async def reset_feedback_data():
    """Manually reset feedback data"""
    try:
        feedback_count = len(feedback_data)
        feedback_data.clear()
        logger.info(f"Manual feedback reset. Cleared {feedback_count} entries.")
        
        return {
            "status": "success",
            "message": f"Reset {feedback_count} feedback entries",
            "current_count": 0
        }
    except Exception as e:
        logger.error(f"Failed to reset feedback data: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset feedback data")

async def monitor_training_progress(training_id: int):
    """Background task to monitor training progress"""
    logger.info(f"Monitoring training progress for job {training_id}")
    
    # Monitor training job status
    max_wait_time = 300  # 5 minutes
    start_time = datetime.now()
    
    while (datetime.now() - start_time).seconds < max_wait_time:
        if training_id in training_jobs:
            job = training_jobs[training_id]
            if job.get('status') in ['completed', 'failed']:
                logger.info(f"Training job {training_id} finished with status: {job.get('status')}")
                break
        
        await asyncio.sleep(5)  # Check every 5 seconds
    
    # Timeout handling
    if training_id in training_jobs and training_jobs[training_id].get('status') == 'running':
        training_jobs[training_id]['status'] = 'failed'
        training_jobs[training_id]['error_message'] = 'Training timeout after 5 minutes'
        logger.warning(f"Training job {training_id} timed out")
