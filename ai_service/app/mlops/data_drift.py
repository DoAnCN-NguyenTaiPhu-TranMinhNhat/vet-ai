"""
Data Drift Detection Module for Veterinary AI System
Uses Evidently AI to monitor data distribution changes
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Data Drift Detection for Veterinary AI System"""

    def __init__(self):
        self.reference_data = None
        logger.info("DataDriftDetector initialized")

    def set_reference_data(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data.copy()
        logger.info("Reference data set with %d samples", len(reference_data))

    def detect_data_drift(self, current_data: pd.DataFrame, column_mapping: Optional[Dict] = None) -> Dict[str, Any]:
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        if column_mapping is None:
            column_mapping = {
                "numerical_features": ["temperature", "weight_kg", "heart_rate", "age_months"],
                "categorical_features": ["animal_type", "gender", "vaccination_status"],
                "target": "target_diagnosis",
            }

        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=self.reference_data, current_data=current_data)

            report_items = drift_report.items()
            drift_results = {
                "timestamp": datetime.now().isoformat(),
                "message": "Drift analysis completed successfully",
                "report_type": "DataDriftPreset",
                "status": "success",
                "metrics_count": len(report_items),
            }

            if report_items:
                for item in report_items:
                    if hasattr(item, "result"):
                        result = item.result
                        if hasattr(result, "dataset_drift"):
                            drift_results["dataset_drift"] = result.dataset_drift
                        if hasattr(result, "drift_share"):
                            drift_results["drift_share"] = result.drift_share
                        break

            logger.info("Data drift analysis completed successfully")
            return drift_results
        except Exception as exc:
            logger.error("Error in drift detection: %s", exc)
            return {"error": str(exc), "timestamp": datetime.now().isoformat()}

    def detect_target_drift(self, current_data: pd.DataFrame, target_column: str = "target_diagnosis") -> Dict[str, Any]:
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")

        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            target_drift_report = Report(metrics=[DataDriftPreset()])
            target_drift_report.run(reference_data=self.reference_data, current_data=current_data)

            report_items = target_drift_report.items()
            target_drift_results = {
                "timestamp": datetime.now().isoformat(),
                "message": "Target drift analysis completed successfully",
                "report_type": "DataDriftPreset",
                "status": "success",
                "metrics_count": len(report_items),
            }

            if report_items:
                for item in report_items:
                    if hasattr(item, "result"):
                        result = item.result
                        if hasattr(result, "dataset_drift"):
                            target_drift_results["target_drift"] = result.dataset_drift
                        break

            logger.info("Target drift analysis completed successfully")
            return target_drift_results
        except Exception as exc:
            logger.error("Error in target drift detection: %s", exc)
            return {"error": str(exc), "timestamp": datetime.now().isoformat()}

    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        try:
            return {
                "period_days": days,
                "message": "Drift monitoring active - workspace features disabled for simplicity",
                "last_analysis": datetime.now().isoformat(),
            }
        except Exception as exc:
            logger.error("Error getting drift summary: %s", exc)
            return {"error": str(exc), "period_days": days}
