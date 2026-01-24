"""
MLOps Integration Module for Veterinary AI System
Integrates data drift detection, model monitoring, and continuous training
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from .data_drift import DataDriftDetector
from .model_monitor import ModelMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLOpsManager:
    """MLOps Manager for Veterinary AI System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MLOps Manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.drift_detector = DataDriftDetector()
        self.model_monitor = ModelMonitor()
        self.reference_data = None
        self.last_drift_check = None
        self.last_health_check = None
        self.training_eligibility = {
            'eligible': False,
            'reason': '',
            'data_quality_score': 0.0,
            'performance_score': 0.0,
            'drift_score': 0.0
        }
        logger.info("MLOpsManager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'drift_threshold': 0.1,  # Drift detection threshold
            'accuracy_threshold': 0.7,  # Minimum accuracy
            'confidence_threshold': 0.6,  # Minimum confidence
            'min_training_samples': 100,  # Minimum samples for retraining
            'max_error_rate': 0.3,  # Maximum error rate
            'check_interval_hours': 1,  # Health check interval
            'retention_days': 30  # Data retention period
        }
    
    def initialize_reference_data(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Initialize reference data for drift detection
        
        Args:
            training_data: Training dataset
            
        Returns:
            Initialization result
        """
        try:
            self.reference_data = training_data.copy()
            self.drift_detector.set_reference_data(training_data)
            
            logger.info(f"Reference data initialized with {len(training_data)} samples")
            return {
                'status': 'success',
                'samples': len(training_data),
                'features': list(training_data.columns),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error initializing reference data: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def process_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a new prediction through MLOps pipeline
        
        Args:
            prediction_data: Prediction data with features
            
        Returns:
            Processing result
        """
        try:
            # Log prediction to model monitor
            monitor_result = self.model_monitor.log_prediction(prediction_data)
            
            # Extract features for drift detection (if available)
            if 'features' in prediction_data:
                features_df = pd.DataFrame([prediction_data['features']])
                
                # Check for data drift periodically
                if self._should_check_drift():
                    drift_result = self.check_data_drift(features_df)
                    if drift_result.get('status') == 'success':
                        logger.info(f"Drift check completed: {drift_result.get('dataset_drift', 'N/A')}")
            
            # Update training eligibility
            self._update_training_eligibility()
            
            return {
                'status': 'success',
                'monitor_result': monitor_result,
                'training_eligible': self.training_eligibility['eligible'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing prediction: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def check_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for data drift
        
        Args:
            current_data: Current data to compare
            
        Returns:
            Drift detection results
        """
        try:
            if self.reference_data is None:
                return {'error': 'Reference data not initialized', 'status': 'failed'}
            
            # Perform drift detection
            drift_result = self.drift_detector.detect_data_drift(current_data)
            self.last_drift_check = datetime.now()
            
            # Update training eligibility based on drift
            if drift_result.get('status') == 'success':
                drift_detected = drift_result.get('dataset_drift', False)
                self.training_eligibility['drift_score'] = 1.0 if drift_detected else 0.0
            
            return drift_result
            
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def get_mlops_status(self) -> Dict[str, Any]:
        """
        Get comprehensive MLOps status
        
        Returns:
            MLOps status dictionary
        """
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'model_metrics': self.model_monitor.get_model_metrics(),
                'model_health': self.model_monitor.check_model_health(),
                'training_eligibility': self.training_eligibility,
                'last_checks': {
                    'drift_check': self.last_drift_check.isoformat() if self.last_drift_check else None,
                    'health_check': self.last_health_check.isoformat() if self.last_health_check else None
                },
                'reference_data': {
                    'initialized': self.reference_data is not None,
                    'samples': len(self.reference_data) if self.reference_data is not None else 0
                }
            }
            
            # Add performance trend
            status['performance_trend'] = self.model_monitor.get_performance_trend(hours=24)
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting MLOps status: {e}")
            return {'error': str(e)}
    
    def should_retrain_model(self) -> Dict[str, Any]:
        """
        Determine if model should be retrained
        
        Returns:
            Retraining recommendation
        """
        try:
            self._update_training_eligibility()
            
            recommendation = {
                'should_retrain': self.training_eligibility['eligible'],
                'reason': self.training_eligibility['reason'],
                'scores': {
                    'data_quality': self.training_eligibility['data_quality_score'],
                    'performance': self.training_eligibility['performance_score'],
                    'drift': self.training_eligibility['drift_score']
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Add additional context
            metrics = self.model_monitor.get_model_metrics()
            recommendation['current_metrics'] = {
                'accuracy': metrics.get('accuracy', 0),
                'error_rate': metrics.get('error_rate', 0),
                'total_predictions': metrics.get('total_predictions', 0)
            }
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error determining retraining need: {e}")
            return {'error': str(e), 'should_retrain': False}
    
    def _update_training_eligibility(self):
        """Update training eligibility based on current metrics"""
        try:
            metrics = self.model_monitor.get_model_metrics()
            health = self.model_monitor.check_model_health()
            
            # Calculate scores (0-1 scale)
            data_quality_score = min(1.0, metrics.get('total_predictions', 0) / self.config['min_training_samples'])
            
            performance_score = 0.0
            accuracy = metrics.get('accuracy', 0)
            if accuracy >= self.config['accuracy_threshold']:
                performance_score = 1.0
            elif accuracy > 0.5:
                performance_score = (accuracy - 0.5) / (self.config['accuracy_threshold'] - 0.5)
            
            error_rate = metrics.get('error_rate', 0)
            if error_rate > self.config['max_error_rate']:
                performance_score *= 0.5  # Penalize high error rate
            
            # Combine scores
            overall_score = (data_quality_score * 0.3 + performance_score * 0.5 + self.training_eligibility['drift_score'] * 0.2)
            
            # Determine eligibility
            eligible = overall_score >= 0.6 and data_quality_score >= 0.5
            
            # Generate reason
            reason = []
            if data_quality_score < 0.5:
                reason.append(f"Insufficient data ({metrics.get('total_predictions', 0)} < {self.config['min_training_samples']})")
            if accuracy < self.config['accuracy_threshold']:
                reason.append(f"Low accuracy ({accuracy:.2f} < {self.config['accuracy_threshold']})")
            if error_rate > self.config['max_error_rate']:
                reason.append(f"High error rate ({error_rate:.2f} > {self.config['max_error_rate']})")
            if self.training_eligibility['drift_score'] > 0.5:
                reason.append("Data drift detected")
            
            if not reason and eligible:
                reason.append("Model performance meets retraining criteria")
            
            self.training_eligibility.update({
                'eligible': eligible,
                'reason': '; '.join(reason) if reason else 'Not eligible',
                'data_quality_score': data_quality_score,
                'performance_score': performance_score,
                'overall_score': overall_score
            })
            
        except Exception as e:
            logger.error(f"Error updating training eligibility: {e}")
    
    def _should_check_drift(self) -> bool:
        """Check if drift detection should run"""
        if self.last_drift_check is None:
            return True
        
        time_since_last = datetime.now() - self.last_drift_check
        return time_since_last.total_seconds() >= (self.config['check_interval_hours'] * 3600)
    
    def export_mlops_data(self, filepath: str = None) -> Dict[str, Any]:
        """
        Export comprehensive MLOps data
        
        Args:
            filepath: Optional file path to save data
            
        Returns:
            Export data dictionary
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'mlops_status': self.get_mlops_status(),
                'retraining_recommendation': self.should_retrain_model(),
                'model_monitoring': self.model_monitor.export_monitoring_data(),
                'config': self.config
            }
            
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                logger.info(f"MLOps data exported to {filepath}")
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting MLOps data: {e}")
            return {'error': str(e)}
    
    def run_health_check(self) -> Dict[str, Any]:
        """
        Run comprehensive health check
        
        Returns:
            Health check results
        """
        try:
            self.last_health_check = datetime.now()
            
            health_results = {
                'timestamp': self.last_health_check.isoformat(),
                'overall_status': 'healthy',
                'checks': []
            }
            
            # Check 1: Model performance
            model_health = self.model_monitor.check_model_health()
            health_results['checks'].append({
                'name': 'Model Performance',
                'status': model_health.get('overall_health', 'unknown'),
                'details': model_health
            })
            
            # Check 2: Data drift detection
            if self.reference_data is not None:
                drift_status = 'ok'
                if self.training_eligibility['drift_score'] > 0.5:
                    drift_status = 'warning'
                health_results['checks'].append({
                    'name': 'Data Drift',
                    'status': drift_status,
                    'drift_score': self.training_eligibility['drift_score']
                })
            else:
                health_results['checks'].append({
                    'name': 'Data Drift',
                    'status': 'not_configured',
                    'message': 'Reference data not initialized'
                })
            
            # Check 3: Training eligibility
            training_status = 'ready' if self.training_eligibility['eligible'] else 'not_ready'
            health_results['checks'].append({
                'name': 'Training Eligibility',
                'status': training_status,
                'eligible': self.training_eligibility['eligible'],
                'overall_score': self.training_eligibility.get('overall_score', 0)
            })
            
            # Determine overall status
            warning_checks = [c for c in health_results['checks'] if c.get('status') in ['warning', 'critical']]
            if warning_checks:
                health_results['overall_status'] = 'warning'
            
            critical_checks = [c for c in health_results['checks'] if c.get('status') == 'critical']
            if critical_checks:
                health_results['overall_status'] = 'critical'
            
            return health_results
            
        except Exception as e:
            logger.error(f"Error running health check: {e}")
            return {'error': str(e), 'overall_status': 'unknown'}

# Example usage and testing
if __name__ == "__main__":
    # Create MLOps Manager
    print("🔧 Testing MLOps Integration")
    print("=" * 50)
    
    mlops = MLOpsManager()
    
    # Initialize with sample reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'temperature': np.random.normal(38.5, 1.0, 1000),
        'weight_kg': np.random.normal(25.0, 5.0, 1000),
        'heart_rate': np.random.normal(80, 10, 1000),
        'age_months': np.random.randint(1, 180, 1000),
        'animal_type': np.random.choice(['Dog', 'Cat', 'Bird'], 1000),
        'gender': np.random.choice(['Male', 'Female'], 1000),
        'vaccination_status': np.random.choice(['Yes', 'No'], 1000),
        'target_diagnosis': np.random.choice(['Healthy', 'Flu', 'Digestive'], 1000)
    })
    
    init_result = mlops.initialize_reference_data(reference_data)
    print(f"Reference Data: {init_result['status']} ({init_result['samples']} samples)")
    
    # Process some predictions
    print("\n📊 Processing Predictions:")
    for i in range(5):
        prediction_data = {
            'prediction': 'Healthy',
            'confidence': 0.85 + (i * 0.02),
            'actual': 'Healthy' if i < 4 else 'Flu',
            'features': {
                'temperature': 38.5,
                'weight_kg': 25.0,
                'heart_rate': 80,
                'age_months': 24,
                'animal_type': 'Dog',
                'gender': 'Male',
                'vaccination_status': 'Yes'
            }
        }
        
        result = mlops.process_prediction(prediction_data)
        print(f"Prediction {i+1}: {result['status']} (Training eligible: {result['training_eligible']})")
    
    # Get MLOps status
    print("\n📈 MLOps Status:")
    status = mlops.get_mlops_status()
    print(f"Overall Health: {status['model_health']['overall_health']}")
    print(f"Training Eligible: {status['training_eligibility']['eligible']}")
    print(f"Total Predictions: {status['model_metrics']['total_predictions']}")
    print(f"Current Accuracy: {status['model_metrics']['accuracy']:.2f}")
    
    # Check retraining recommendation
    print("\n🔄 Retraining Recommendation:")
    retrain = mlops.should_retrain_model()
    print(f"Should Retrain: {retrain['should_retrain']}")
    print(f"Reason: {retrain['reason']}")
    print(f"Overall Score: {retrain['scores'].get('overall_score', 0):.2f}")
    
    # Run health check
    print("\n🏥 Health Check:")
    health = mlops.run_health_check()
    print(f"Overall Status: {health['overall_status']}")
    for check in health['checks']:
        print(f"  - {check['name']}: {check['status']}")
    
    print("\n✅ MLOps Integration Working Successfully!")
