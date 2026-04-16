"""
MLOps Integration Module for Veterinary AI System
Integrates data drift detection, model monitoring, and continuous training
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from ai_service.app.mlops.data_drift import DataDriftDetector
from ai_service.app.mlops.monitor import ModelMonitor

logger = logging.getLogger(__name__)


class MLOpsManager:
    """MLOps Manager for Veterinary AI System"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.drift_detector = DataDriftDetector()
        self.model_monitor = ModelMonitor()
        self.reference_data = None
        self.last_drift_check = None
        self.last_health_check = None
        self.training_eligibility = {
            "eligible": False,
            "reason": "",
            "data_quality_score": 0.0,
            "performance_score": 0.0,
            "drift_score": 0.0,
        }
        logger.info("MLOpsManager initialized")

    def _default_config(self) -> Dict[str, Any]:
        return {
            "drift_threshold": 0.1,
            "accuracy_threshold": 0.7,
            "confidence_threshold": 0.6,
            "min_training_samples": 100,
            "max_error_rate": 0.3,
            "check_interval_hours": 1,
            "retention_days": 30,
        }

    def initialize_reference_data(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            self.reference_data = training_data.copy()
            self.drift_detector.set_reference_data(training_data)
            logger.info("Reference data initialized with %d samples", len(training_data))
            return {
                "status": "success",
                "samples": len(training_data),
                "features": list(training_data.columns),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            logger.error("Error initializing reference data: %s", exc)
            return {"error": str(exc), "status": "failed"}

    def process_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            monitor_result = self.model_monitor.log_prediction(prediction_data)
            if "features" in prediction_data:
                features_df = pd.DataFrame([prediction_data["features"]])
                if self._should_check_drift():
                    drift_result = self.check_data_drift(features_df)
                    if drift_result.get("status") == "success":
                        logger.info("Drift check completed: %s", drift_result.get("dataset_drift", "N/A"))

            self._update_training_eligibility()
            return {
                "status": "success",
                "monitor_result": monitor_result,
                "training_eligible": self.training_eligibility["eligible"],
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            logger.error("Error processing prediction: %s", exc)
            return {"error": str(exc), "status": "failed"}

    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if self.reference_data is None:
                return {"error": "Reference data not initialized", "status": "failed"}
            drift_result = self.drift_detector.detect_data_drift(current_data)
            self.last_drift_check = datetime.now()
            if drift_result.get("status") == "success":
                self.training_eligibility["drift_score"] = 1.0 if drift_result.get("dataset_drift", False) else 0.0
            return drift_result
        except Exception as exc:
            logger.error("Error checking data drift: %s", exc)
            return {"error": str(exc), "status": "failed"}

    def get_mlops_status(self) -> Dict[str, Any]:
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "config": self.config,
                "model_metrics": self.model_monitor.get_model_metrics(),
                "model_health": self.model_monitor.check_model_health(),
                "training_eligibility": self.training_eligibility,
                "last_checks": {
                    "drift_check": self.last_drift_check.isoformat() if self.last_drift_check else None,
                    "health_check": self.last_health_check.isoformat() if self.last_health_check else None,
                },
                "reference_data": {
                    "initialized": self.reference_data is not None,
                    "samples": len(self.reference_data) if self.reference_data is not None else 0,
                },
            }
            status["performance_trend"] = self.model_monitor.get_performance_trend(hours=24)
            return status
        except Exception as exc:
            logger.error("Error getting MLOps status: %s", exc)
            return {"error": str(exc)}

    def should_retrain_model(self) -> Dict[str, Any]:
        try:
            self._update_training_eligibility()
            metrics = self.model_monitor.get_model_metrics()
            return {
                "should_retrain": self.training_eligibility["eligible"],
                "reason": self.training_eligibility["reason"],
                "scores": {
                    "data_quality": self.training_eligibility["data_quality_score"],
                    "performance": self.training_eligibility["performance_score"],
                    "drift": self.training_eligibility["drift_score"],
                },
                "timestamp": datetime.now().isoformat(),
                "current_metrics": {
                    "accuracy": metrics.get("accuracy", 0),
                    "error_rate": metrics.get("error_rate", 0),
                    "total_predictions": metrics.get("total_predictions", 0),
                },
            }
        except Exception as exc:
            logger.error("Error determining retraining need: %s", exc)
            return {"error": str(exc), "should_retrain": False}

    def _update_training_eligibility(self):
        try:
            metrics = self.model_monitor.get_model_metrics()
            data_quality_score = min(1.0, metrics.get("total_predictions", 0) / self.config["min_training_samples"])

            performance_score = 0.0
            accuracy = metrics.get("accuracy", 0)
            if accuracy >= self.config["accuracy_threshold"]:
                performance_score = 1.0
            elif accuracy > 0.5:
                performance_score = (accuracy - 0.5) / (self.config["accuracy_threshold"] - 0.5)

            error_rate = metrics.get("error_rate", 0)
            if error_rate > self.config["max_error_rate"]:
                performance_score *= 0.5

            overall_score = (
                data_quality_score * 0.3
                + performance_score * 0.5
                + self.training_eligibility["drift_score"] * 0.2
            )
            eligible = overall_score >= 0.6 and data_quality_score >= 0.5

            reason = []
            if data_quality_score < 0.5:
                reason.append(
                    f"Insufficient data ({metrics.get('total_predictions', 0)} < {self.config['min_training_samples']})"
                )
            if accuracy < self.config["accuracy_threshold"]:
                reason.append(f"Low accuracy ({accuracy:.2f} < {self.config['accuracy_threshold']})")
            if error_rate > self.config["max_error_rate"]:
                reason.append(f"High error rate ({error_rate:.2f} > {self.config['max_error_rate']})")
            if self.training_eligibility["drift_score"] > 0.5:
                reason.append("Data drift detected")
            if not reason and eligible:
                reason.append("Model performance meets retraining criteria")

            self.training_eligibility.update(
                {
                    "eligible": eligible,
                    "reason": "; ".join(reason) if reason else "Not eligible",
                    "data_quality_score": data_quality_score,
                    "performance_score": performance_score,
                    "overall_score": overall_score,
                }
            )
        except Exception as exc:
            logger.error("Error updating training eligibility: %s", exc)

    def _should_check_drift(self) -> bool:
        if self.last_drift_check is None:
            return True
        time_since_last = datetime.now() - self.last_drift_check
        return time_since_last.total_seconds() >= (self.config["check_interval_hours"] * 3600)

    def export_mlops_data(self, filepath: str = None) -> Dict[str, Any]:
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "mlops_status": self.get_mlops_status(),
                "retraining_recommendation": self.should_retrain_model(),
                "model_monitoring": self.model_monitor.export_monitoring_data(),
                "config": self.config,
            }
            if filepath:
                with open(filepath, "w", encoding="utf-8") as handle:
                    json.dump(export_data, handle, indent=2)
                logger.info("MLOps data exported to %s", filepath)
            return export_data
        except Exception as exc:
            logger.error("Error exporting MLOps data: %s", exc)
            return {"error": str(exc)}

    def run_health_check(self) -> Dict[str, Any]:
        try:
            self.last_health_check = datetime.now()
            health_results = {"timestamp": self.last_health_check.isoformat(), "overall_status": "healthy", "checks": []}
            model_health = self.model_monitor.check_model_health()
            health_results["checks"].append(
                {"name": "Model Performance", "status": model_health.get("overall_health", "unknown"), "details": model_health}
            )
            if self.reference_data is not None:
                drift_status = "warning" if self.training_eligibility["drift_score"] > 0.5 else "ok"
                health_results["checks"].append(
                    {"name": "Data Drift", "status": drift_status, "drift_score": self.training_eligibility["drift_score"]}
                )
            else:
                health_results["checks"].append(
                    {"name": "Data Drift", "status": "not_configured", "message": "Reference data not initialized"}
                )

            health_results["checks"].append(
                {
                    "name": "Training Eligibility",
                    "status": "ready" if self.training_eligibility["eligible"] else "not_ready",
                    "eligible": self.training_eligibility["eligible"],
                    "overall_score": self.training_eligibility.get("overall_score", 0),
                }
            )

            if [c for c in health_results["checks"] if c.get("status") in ["warning", "critical"]]:
                health_results["overall_status"] = "warning"
            if [c for c in health_results["checks"] if c.get("status") == "critical"]:
                health_results["overall_status"] = "critical"
            return health_results
        except Exception as exc:
            logger.error("Error running health check: %s", exc)
            return {"error": str(exc), "overall_status": "unknown"}
