"""
MLOps Champion-Challenger Strategy Implementation
Handles model promotion, evaluation, and rollback with manual approval
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class ModelStatus(Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"

class ChampionChallengerManager:
    """Manages Champion-Challenger MLOps workflow"""
    
    def __init__(self, 
                 model_registry_path: str = "/app/models",
                 mlflow_tracking_uri: str = None,
                 approval_required: bool = True):
        
        self.model_registry_path = model_registry_path
        self.approval_required = approval_required
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        
        # Initialize MLflow client
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.client = MlflowClient()
        
        # Thresholds for evaluation
        self.accuracy_gain_threshold = float(os.getenv("ACCURACY_GAIN_THRESHOLD", "0.01"))
        self.f1_score_min = float(os.getenv("F1_SCORE_MIN", "0.85"))
        
        # Ensure registry directories exist
        os.makedirs(f"{model_registry_path}/staging", exist_ok=True)
        os.makedirs(f"{model_registry_path}/production", exist_ok=True)
        os.makedirs(f"{model_registry_path}/archived", exist_ok=True)
        
    def register_staging_model(self, 
                              model_path: str, 
                              training_metrics: Dict,
                              model_version: str) -> Dict:
        """Register new trained model to STAGING environment"""
        
        try:
            staging_path = f"{self.model_registry_path}/staging/{model_version}"
            os.makedirs(staging_path, exist_ok=True)
            
            # Copy model to staging
            import shutil
            if os.path.isdir(model_path):
                shutil.copytree(model_path, staging_path, dirs_exist_ok=True)
            else:
                shutil.copy2(model_path, staging_path)
            
            # Save staging metadata
            staging_metadata = {
                "model_version": model_version,
                "status": ModelStatus.STAGING.value,
                "registered_at": datetime.now().isoformat(),
                "training_metrics": training_metrics,
                "evaluation_metrics": None,
                "approval_status": "pending",
                "promoted_to_production": None
            }
            
            with open(f"{staging_path}/staging_metadata.json", "w") as f:
                json.dump(staging_metadata, f, indent=2)
            
            # Log to MLflow
            if self.mlflow_tracking_uri:
                with mlflow.start_run(run_name=f"staging-{model_version}"):
                    mlflow.log_metrics(training_metrics)
                    mlflow.log_artifacts(staging_path, "staging_model")
                    mlflow.set_tag("stage", "staging")
            
            logger.info(f"Model {model_version} registered to STAGING")
            return staging_metadata
            
        except Exception as e:
            logger.error(f"Failed to register staging model: {e}")
            raise
    
    def evaluate_staging_model(self, 
                              staging_model_version: str,
                              test_data: pd.DataFrame = None) -> Dict:
        """Evaluate staging model against current production model"""
        
        try:
            # Get current production model
            production_model_path = self._get_current_production_model()
            if not production_model_path:
                logger.warning("No production model found for comparison")
                return {"error": "No production model available"}
            
            # Load models
            staging_model = self._load_model(f"{self.model_registry_path}/staging/{staging_model_version}")
            production_model = self._load_model(production_model_path)
            
            # Evaluate on test data
            if test_data is None:
                # Generate synthetic test data for demo
                test_data = self._generate_test_data()
            
            X_test = test_data.drop('target', axis=1)
            y_test = test_data['target']
            
            # Get predictions
            staging_preds = staging_model.predict(X_test)
            production_preds = production_model.predict(X_test)
            
            # Calculate metrics
            staging_metrics = {
                "accuracy": accuracy_score(y_test, staging_preds),
                "f1_score": f1_score(y_test, staging_preds, average='weighted')
            }
            
            production_metrics = {
                "accuracy": accuracy_score(y_test, production_preds),
                "f1_score": f1_score(y_test, production_preds, average='weighted')
            }
            
            # Calculate gains
            accuracy_gain = staging_metrics["accuracy"] - production_metrics["accuracy"]
            f1_gain = staging_metrics["f1_score"] - production_metrics["f1_score"]
            
            evaluation_result = {
                "staging_model_version": staging_model_version,
                "production_model_path": production_model_path,
                "staging_metrics": staging_metrics,
                "production_metrics": production_metrics,
                "gains": {
                    "accuracy_gain": accuracy_gain,
                    "f1_gain": f1_gain
                },
                "passes_threshold": {
                    "accuracy_gain_ok": accuracy_gain > self.accuracy_gain_threshold,
                    "f1_score_min_ok": staging_metrics["f1_score"] >= self.f1_score_min
                },
                "recommendation": self._get_recommendation(accuracy_gain, staging_metrics["f1_score"]),
                "evaluated_at": datetime.now().isoformat()
            }
            
            # Save evaluation results
            eval_path = f"{self.model_registry_path}/staging/{staging_model_version}/evaluation.json"
            with open(eval_path, "w") as f:
                json.dump(evaluation_result, f, indent=2)
            
            # Log to MLflow
            if self.mlflow_tracking_uri:
                with mlflow.start_run(run_name=f"evaluation-{staging_model_version}"):
                    mlflow.log_metrics({
                        "staging_accuracy": staging_metrics["accuracy"],
                        "staging_f1": staging_metrics["f1_score"],
                        "production_accuracy": production_metrics["accuracy"],
                        "production_f1": production_metrics["f1_score"],
                        "accuracy_gain": accuracy_gain,
                        "f1_gain": f1_gain
                    })
                    mlflow.log_artifacts(eval_path, "evaluation")
            
            logger.info(f"Evaluation completed for {staging_model_version}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
    
    def request_approval(self, staging_model_version: str, 
                        requestor: str = "system") -> Dict:
        """Create approval request for model promotion"""
        
        try:
            # Load evaluation results
            eval_path = f"{self.model_registry_path}/staging/{staging_model_version}/evaluation.json"
            if not os.path.exists(eval_path):
                raise FileNotFoundError("Model not evaluated yet")
            
            with open(eval_path, "r") as f:
                evaluation = json.load(f)
            
            # Create approval request
            approval_request = {
                "model_version": staging_model_version,
                "requestor": requestor,
                "requested_at": datetime.now().isoformat(),
                "evaluation_summary": evaluation,
                "status": "pending_approval",
                "approver": None,
                "approved_at": None,
                "comments": None
            }
            
            # Save approval request
            approval_path = f"{self.model_registry_path}/staging/{staging_model_version}/approval.json"
            with open(approval_path, "w") as f:
                json.dump(approval_request, f, indent=2)
            
            logger.info(f"Approval request created for {staging_model_version}")
            return approval_request
            
        except Exception as e:
            logger.error(f"Failed to create approval request: {e}")
            raise
    
    def approve_model_promotion(self, 
                              staging_model_version: str,
                              approver: str,
                              comments: str = None) -> Dict:
        """Approve and promote staging model to production"""
        
        try:
            # Load approval request
            approval_path = f"{self.model_registry_path}/staging/{staging_model_version}/approval.json"
            with open(approval_path, "r") as f:
                approval = json.load(f)
            
            # Update approval
            approval["status"] = "approved"
            approval["approver"] = approver
            approval["approved_at"] = datetime.now().isoformat()
            approval["comments"] = comments
            
            with open(approval_path, "w") as f:
                json.dump(approval, f, indent=2)
            
            # Archive current production model
            self._archive_current_production()
            
            # Promote staging to production
            staging_path = f"{self.model_registry_path}/staging/{staging_model_version}"
            production_path = f"{self.model_registry_path}/production/{staging_model_version}"
            
            import shutil
            shutil.copytree(staging_path, production_path)
            
            # Update production metadata
            prod_metadata = {
                "model_version": staging_model_version,
                "status": ModelStatus.PRODUCTION.value,
                "promoted_at": datetime.now().isoformat(),
                "promoted_by": approver,
                "approval": approval
            }
            
            with open(f"{production_path}/production_metadata.json", "w") as f:
                json.dump(prod_metadata, f, indent=2)
            
            # Log to MLflow
            if self.mlflow_tracking_uri:
                with mlflow.start_run(run_name=f"promotion-{staging_model_version}"):
                    mlflow.log_metrics({"promotion_status": 1})
                    mlflow.set_tag("promoted_by", approver)
                    mlflow.set_tag("stage", "production")
            
            logger.info(f"Model {staging_model_version} promoted to PRODUCTION by {approver}")
            return prod_metadata
            
        except Exception as e:
            logger.error(f"Failed to approve model promotion: {e}")
            raise
    
    def reject_model_promotion(self, 
                              staging_model_version: str,
                              approver: str,
                              reason: str) -> Dict:
        """Reject staging model promotion"""
        
        try:
            approval_path = f"{self.model_registry_path}/staging/{staging_model_version}/approval.json"
            with open(approval_path, "r") as f:
                approval = json.load(f)
            
            approval["status"] = "rejected"
            approval["approver"] = approver
            approval["approved_at"] = datetime.now().isoformat()
            approval["comments"] = reason
            
            with open(approval_path, "w") as f:
                json.dump(approval, f, indent=2)
            
            # Move to failed after 7 days
            self._schedule_cleanup(staging_model_version)
            
            logger.info(f"Model {staging_model_version} rejected by {approver}")
            return approval
            
        except Exception as e:
            logger.error(f"Failed to reject model: {e}")
            raise
    
    def rollback_to_archived(self, archived_model_version: str) -> Dict:
        """Rollback to an archived model version"""
        
        try:
            # Archive current production
            self._archive_current_production()
            
            # Restore archived model
            archived_path = f"{self.model_registry_path}/archived/{archived_model_version}"
            production_path = f"{self.model_registry_path}/production/{archived_model_version}"
            
            import shutil
            shutil.copytree(archived_path, production_path)
            
            rollback_metadata = {
                "rollback_from": archived_model_version,
                "rolled_back_at": datetime.now().isoformat(),
                "rollback_reason": "Manual rollback request"
            }
            
            with open(f"{production_path}/rollback_metadata.json", "w") as f:
                json.dump(rollback_metadata, f, indent=2)
            
            logger.info(f"Rollback completed to {archived_model_version}")
            return rollback_metadata
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise
    
    def get_model_registry_status(self) -> Dict:
        """Get current status of all models in registry"""
        
        status = {
            "production": self._list_models_by_status(ModelStatus.PRODUCTION),
            "staging": self._list_models_by_status(ModelStatus.STAGING),
            "archived": self._list_models_by_status(ModelStatus.ARCHIVED),
            "failed": self._list_models_by_status(ModelStatus.FAILED)
        }
        
        return status
    
    # Helper methods
    def _get_current_production_model(self) -> Optional[str]:
        """Get path to current production model"""
        prod_dir = f"{self.model_registry_path}/production"
        if not os.path.exists(prod_dir):
            return None
        
        models = [d for d in os.listdir(prod_dir) 
                 if os.path.isdir(os.path.join(prod_dir, d))]
        
        if not models:
            return None
        
        # Return the most recent model
        latest_model = sorted(models, reverse=True)[0]
        return os.path.join(prod_dir, latest_model)
    
    def _load_model(self, model_path: str):
        """Load model from path"""
        import joblib
        model_file = os.path.join(model_path, "model.pkl")
        if os.path.exists(model_file):
            return joblib.load(model_file)
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
    
    def _generate_test_data(self) -> pd.DataFrame:
        """Generate synthetic test data for evaluation"""
        import numpy as np
        np.random.seed(42)
        
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 3, n_samples)  # 3 classes
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    def _get_recommendation(self, accuracy_gain: float, f1_score: float) -> str:
        """Get promotion recommendation based on metrics"""
        
        if accuracy_gain > self.accuracy_gain_threshold and f1_score >= self.f1_score_min:
            return "APPROVE"
        elif f1_score >= self.f1_score_min:
            return "CONSIDER"
        else:
            return "REJECT"
    
    def _archive_current_production(self):
        """Archive current production model"""
        current_prod = self._get_current_production_model()
        if current_prod:
            import shutil
            archived_path = current_prod.replace("/production/", "/archived/")
            shutil.copytree(current_prod, archived_path)
    
    def _schedule_cleanup(self, model_version: str):
        """Schedule cleanup of failed models after 7 days"""
        cleanup_date = datetime.now() + timedelta(days=7)
        cleanup_info = {
            "model_version": model_version,
            "cleanup_date": cleanup_date.isoformat(),
            "reason": "Failed evaluation"
        }
        
        cleanup_path = f"{self.model_registry_path}/staging/{model_version}/cleanup.json"
        with open(cleanup_path, "w") as f:
            json.dump(cleanup_info, f, indent=2)
    
    def _list_models_by_status(self, status: ModelStatus) -> List[Dict]:
        """List models by their status"""
        status_dir = f"{self.model_registry_path}/{status.value}"
        if not os.path.exists(status_dir):
            return []
        
        models = []
        for model_name in os.listdir(status_dir):
            model_path = os.path.join(status_dir, model_name)
            if os.path.isdir(model_path):
                models.append({
                    "version": model_name,
                    "path": model_path,
                    "status": status.value
                })
        
        return sorted(models, key=lambda x: x["version"], reverse=True)
