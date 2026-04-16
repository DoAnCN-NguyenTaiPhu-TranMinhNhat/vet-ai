"""
Model Performance Monitoring Module for Veterinary AI System
Tracks model metrics, predictions, and performance over time
"""

import json
import logging
import statistics
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Model Performance Monitoring for Veterinary AI System"""

    def __init__(self, max_history_days: int = 30):
        self.max_history_days = max_history_days
        self.predictions_history = deque(maxlen=10000)
        self.performance_metrics = defaultdict(list)
        self.alerts = []
        self.model_metrics = {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "avg_confidence": 0.0,
            "predictions_per_hour": 0,
            "error_rate": 0.0,
            "last_updated": datetime.now().isoformat(),
        }
        logger.info("ModelMonitor initialized")

    def log_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if "timestamp" not in prediction_data:
                prediction_data["timestamp"] = datetime.now().isoformat()

            for field in ("prediction", "confidence"):
                if field not in prediction_data:
                    raise ValueError(f"Missing required field: {field}")

            self.predictions_history.append(prediction_data)
            self.model_metrics["total_predictions"] += 1
            self.model_metrics["last_updated"] = datetime.now().isoformat()

            if "actual" in prediction_data:
                if prediction_data["prediction"] == prediction_data["actual"]:
                    self.model_metrics["correct_predictions"] += 1
                total = self.model_metrics["total_predictions"]
                correct = self.model_metrics["correct_predictions"]
                self.model_metrics["accuracy"] = correct / total if total > 0 else 0.0

            confidences = [p["confidence"] for p in self.predictions_history]
            self.model_metrics["avg_confidence"] = statistics.mean(confidences) if confidences else 0.0
            recent_predictions = [p for p in self.predictions_history if self._is_recent(p["timestamp"], hours=1)]
            self.model_metrics["predictions_per_hour"] = len(recent_predictions)

            error_count = len(
                [p for p in self.predictions_history if "actual" in p and p["prediction"] != p["actual"]]
            )
            total_with_actual = len([p for p in self.predictions_history if "actual" in p])
            self.model_metrics["error_rate"] = error_count / total_with_actual if total_with_actual > 0 else 0.0

            self._check_performance_alerts(prediction_data)
            logger.info(
                "Prediction logged: %s (confidence: %.2f)",
                prediction_data["prediction"],
                prediction_data["confidence"],
            )
            return {"status": "success", "prediction_id": len(self.predictions_history)}
        except Exception as exc:
            logger.error("Error logging prediction: %s", exc)
            return {"error": str(exc), "status": "failed"}

    def get_model_metrics(self) -> Dict[str, Any]:
        try:
            metrics = self.model_metrics.copy()
            confidences = [p["confidence"] for p in self.predictions_history]
            if confidences:
                metrics["confidence_stats"] = {
                    "min": min(confidences),
                    "max": max(confidences),
                    "median": statistics.median(confidences),
                    "std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
                }

            prediction_counts = defaultdict(int)
            for prediction in self.predictions_history:
                prediction_counts[prediction["prediction"]] += 1
            metrics["prediction_distribution"] = dict(prediction_counts)

            recent_predictions = [p for p in self.predictions_history if self._is_recent(p["timestamp"], hours=24)]
            if recent_predictions:
                recent_correct = len(
                    [p for p in recent_predictions if "actual" in p and p["prediction"] == p["actual"]]
                )
                recent_total = len([p for p in recent_predictions if "actual" in p])
                metrics["recent_24h_accuracy"] = recent_correct / recent_total if recent_total > 0 else 0.0
                metrics["recent_24h_predictions"] = len(recent_predictions)

            metrics["active_alerts"] = len([a for a in self.alerts if not a.get("resolved", False)])
            metrics["total_alerts"] = len(self.alerts)
            return metrics
        except Exception as exc:
            logger.error("Error getting model metrics: %s", exc)
            return {"error": str(exc)}

    def get_performance_trend(self, hours: int = 24) -> Dict[str, Any]:
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_predictions = [
                p for p in self.predictions_history if datetime.fromisoformat(p["timestamp"].replace("Z", "+00:00")) > cutoff_time
            ]
            if not recent_predictions:
                return {"message": f"No predictions in last {hours} hours", "trend": []}

            hourly_data = defaultdict(list)
            for prediction in recent_predictions:
                hour_key = datetime.fromisoformat(prediction["timestamp"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:00")
                hourly_data[hour_key].append(prediction)

            trend = []
            for hour, predictions in sorted(hourly_data.items()):
                correct = len([p for p in predictions if "actual" in p and p["prediction"] == p["actual"]])
                total_with_actual = len([p for p in predictions if "actual" in p])
                accuracy = correct / total_with_actual if total_with_actual > 0 else 0.0
                avg_confidence = statistics.mean([p["confidence"] for p in predictions]) if predictions else 0.0
                trend.append(
                    {
                        "hour": hour,
                        "predictions": len(predictions),
                        "accuracy": accuracy,
                        "avg_confidence": avg_confidence,
                        "error_rate": 1.0 - accuracy,
                    }
                )

            return {"period_hours": hours, "total_predictions": len(recent_predictions), "trend": trend}
        except Exception as exc:
            logger.error("Error getting performance trend: %s", exc)
            return {"error": str(exc)}

    def check_model_health(self) -> Dict[str, Any]:
        try:
            health_status = {"overall_health": "healthy", "checks": [], "alerts": [], "recommendations": []}
            if self.model_metrics["accuracy"] < 0.7:
                health_status["checks"].append(
                    {
                        "name": "Accuracy Check",
                        "status": "warning",
                        "message": f"Accuracy {self.model_metrics['accuracy']:.2f} below threshold 0.7",
                    }
                )
                health_status["recommendations"].append("Consider retraining model with more data")

            if self.model_metrics["avg_confidence"] < 0.6:
                health_status["checks"].append(
                    {
                        "name": "Confidence Check",
                        "status": "warning",
                        "message": f"Average confidence {self.model_metrics['avg_confidence']:.2f} below threshold 0.6",
                    }
                )
                health_status["recommendations"].append("Model may be uncertain - review training data")

            if self.model_metrics["predictions_per_hour"] < 1:
                health_status["checks"].append(
                    {
                        "name": "Prediction Volume",
                        "status": "info",
                        "message": f"Low prediction volume: {self.model_metrics['predictions_per_hour']:.1f} per hour",
                    }
                )

            if self.model_metrics["error_rate"] > 0.3:
                health_status["checks"].append(
                    {
                        "name": "Error Rate",
                        "status": "critical",
                        "message": f"High error rate: {self.model_metrics['error_rate']:.2f}",
                    }
                )
                health_status["recommendations"].append("Immediate model retraining recommended")
                health_status["overall_health"] = "critical"
            elif self.model_metrics["error_rate"] > 0.2:
                health_status["checks"].append(
                    {
                        "name": "Error Rate",
                        "status": "warning",
                        "message": f"Elevated error rate: {self.model_metrics['error_rate']:.2f}",
                    }
                )
                if health_status["overall_health"] == "healthy":
                    health_status["overall_health"] = "warning"

            health_status["alerts"] = [a for a in self.alerts if not a.get("resolved", False)]
            return health_status
        except Exception as exc:
            logger.error("Error checking model health: %s", exc)
            return {"error": str(exc), "overall_health": "unknown"}

    def _is_recent(self, timestamp: str, hours: int) -> bool:
        try:
            prediction_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return prediction_time > cutoff_time
        except Exception:
            return False

    def _check_performance_alerts(self, prediction_data: Dict[str, Any]):
        try:
            if prediction_data["confidence"] < 0.3:
                self.alerts.append(
                    {
                        "type": "low_confidence",
                        "message": f"Very low confidence: {prediction_data['confidence']:.2f}",
                        "timestamp": datetime.now().isoformat(),
                        "prediction": prediction_data["prediction"],
                        "resolved": False,
                    }
                )

            if "actual" in prediction_data and prediction_data["prediction"] != prediction_data["actual"]:
                recent_errors = [
                    p
                    for p in list(self.predictions_history)[-10:]
                    if "actual" in p and p["prediction"] != p["actual"]
                ]
                if len(recent_errors) >= 5:
                    self.alerts.append(
                        {
                            "type": "high_error_rate",
                            "message": f"High error rate detected: {len(recent_errors)}/10 recent predictions",
                            "timestamp": datetime.now().isoformat(),
                            "resolved": False,
                        }
                    )
        except Exception as exc:
            logger.error("Error checking alerts: %s", exc)

    def export_monitoring_data(self, filepath: str = None) -> Dict[str, Any]:
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "model_metrics": self.get_model_metrics(),
                "performance_trend": self.get_performance_trend(hours=24),
                "health_check": self.check_model_health(),
                "recent_predictions": list(self.predictions_history)[-100:],
                "alerts": self.alerts[-50:],
            }
            if filepath:
                with open(filepath, "w", encoding="utf-8") as handle:
                    json.dump(export_data, handle, indent=2)
                logger.info("Monitoring data exported to %s", filepath)
            return export_data
        except Exception as exc:
            logger.error("Error exporting monitoring data: %s", exc)
            return {"error": str(exc)}
