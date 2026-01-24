"""
Model Performance Monitoring Module for Veterinary AI System
Tracks model metrics, predictions, and performance over time
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Model Performance Monitoring for Veterinary AI System"""
    
    def __init__(self, max_history_days: int = 30):
        """
        Initialize model monitor
        
        Args:
            max_history_days: Maximum days to keep monitoring data
        """
        self.max_history_days = max_history_days
        self.predictions_history = deque(maxlen=10000)  # Store recent predictions
        self.performance_metrics = defaultdict(list)
        self.alerts = []
        self.model_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'predictions_per_hour': 0,
            'error_rate': 0.0,
            'last_updated': datetime.now().isoformat()
        }
        logger.info("ModelMonitor initialized")
    
    def log_prediction(self, prediction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Log a single prediction for monitoring
        
        Args:
            prediction_data: Dictionary containing prediction info
                - prediction: predicted class
                - confidence: prediction confidence
                - actual: actual class (if available)
                - timestamp: prediction timestamp
                - features: input features (optional)
                
        Returns:
            Monitoring result
        """
        try:
            # Add timestamp if not provided
            if 'timestamp' not in prediction_data:
                prediction_data['timestamp'] = datetime.now().isoformat()
            
            # Validate required fields
            required_fields = ['prediction', 'confidence']
            for field in required_fields:
                if field not in prediction_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Add to history
            self.predictions_history.append(prediction_data)
            
            # Update metrics
            self.model_metrics['total_predictions'] += 1
            self.model_metrics['last_updated'] = datetime.now().isoformat()
            
            # Calculate accuracy if actual label available
            if 'actual' in prediction_data:
                if prediction_data['prediction'] == prediction_data['actual']:
                    self.model_metrics['correct_predictions'] += 1
                
                # Update accuracy
                total = self.model_metrics['total_predictions']
                correct = self.model_metrics['correct_predictions']
                self.model_metrics['accuracy'] = correct / total if total > 0 else 0.0
            
            # Update average confidence
            confidences = [p['confidence'] for p in self.predictions_history]
            self.model_metrics['avg_confidence'] = statistics.mean(confidences) if confidences else 0.0
            
            # Calculate predictions per hour
            recent_predictions = [p for p in self.predictions_history 
                                if self._is_recent(p['timestamp'], hours=1)]
            self.model_metrics['predictions_per_hour'] = len(recent_predictions)
            
            # Calculate error rate
            error_count = len([p for p in self.predictions_history 
                            if 'actual' in p and p['prediction'] != p['actual']])
            total_with_actual = len([p for p in self.predictions_history if 'actual' in p])
            self.model_metrics['error_rate'] = error_count / total_with_actual if total_with_actual > 0 else 0.0
            
            # Check for alerts
            self._check_performance_alerts(prediction_data)
            
            logger.info(f"Prediction logged: {prediction_data['prediction']} (confidence: {prediction_data['confidence']:.2f})")
            return {'status': 'success', 'prediction_id': len(self.predictions_history)}
            
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get current model performance metrics
        
        Returns:
            Dictionary of model metrics
        """
        try:
            # Calculate additional metrics
            metrics = self.model_metrics.copy()
            
            # Add confidence distribution
            confidences = [p['confidence'] for p in self.predictions_history]
            if confidences:
                metrics['confidence_stats'] = {
                    'min': min(confidences),
                    'max': max(confidences),
                    'median': statistics.median(confidences),
                    'std': statistics.stdev(confidences) if len(confidences) > 1 else 0.0
                }
            
            # Add prediction distribution by class
            prediction_counts = defaultdict(int)
            for p in self.predictions_history:
                prediction_counts[p['prediction']] += 1
            
            metrics['prediction_distribution'] = dict(prediction_counts)
            
            # Add recent performance (last 24 hours)
            recent_predictions = [p for p in self.predictions_history 
                                if self._is_recent(p['timestamp'], hours=24)]
            
            if recent_predictions:
                recent_correct = len([p for p in recent_predictions 
                                    if 'actual' in p and p['prediction'] == p['actual']])
                recent_total = len([p for p in recent_predictions if 'actual' in p])
                metrics['recent_24h_accuracy'] = recent_correct / recent_total if recent_total > 0 else 0.0
                metrics['recent_24h_predictions'] = len(recent_predictions)
            
            # Add alerts summary
            metrics['active_alerts'] = len([a for a in self.alerts if not a.get('resolved', False)])
            metrics['total_alerts'] = len(self.alerts)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {'error': str(e)}
    
    def get_performance_trend(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance trend over specified time period
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Performance trend data
        """
        try:
            # Filter predictions by time period
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_predictions = [p for p in self.predictions_history 
                                if datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')) > cutoff_time]
            
            if not recent_predictions:
                return {'message': f'No predictions in last {hours} hours', 'trend': []}
            
            # Group by hour
            hourly_data = defaultdict(list)
            for p in recent_predictions:
                hour_key = datetime.fromisoformat(p['timestamp'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:00')
                hourly_data[hour_key].append(p)
            
            # Calculate hourly metrics
            trend = []
            for hour, predictions in sorted(hourly_data.items()):
                correct = len([p for p in predictions if 'actual' in p and p['prediction'] == p['actual']])
                total_with_actual = len([p for p in predictions if 'actual' in p])
                accuracy = correct / total_with_actual if total_with_actual > 0 else 0.0
                
                avg_confidence = statistics.mean([p['confidence'] for p in predictions]) if predictions else 0.0
                
                trend.append({
                    'hour': hour,
                    'predictions': len(predictions),
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'error_rate': 1.0 - accuracy
                })
            
            return {
                'period_hours': hours,
                'total_predictions': len(recent_predictions),
                'trend': trend
            }
            
        except Exception as e:
            logger.error(f"Error getting performance trend: {e}")
            return {'error': str(e)}
    
    def check_model_health(self) -> Dict[str, Any]:
        """
        Check overall model health and generate alerts
        
        Returns:
            Health check results
        """
        try:
            health_status = {
                'overall_health': 'healthy',
                'checks': [],
                'alerts': [],
                'recommendations': []
            }
            
            # Check 1: Accuracy threshold
            if self.model_metrics['accuracy'] < 0.7:
                health_status['checks'].append({
                    'name': 'Accuracy Check',
                    'status': 'warning',
                    'message': f"Accuracy {self.model_metrics['accuracy']:.2f} below threshold 0.7"
                })
                health_status['recommendations'].append("Consider retraining model with more data")
            
            # Check 2: Confidence threshold
            if self.model_metrics['avg_confidence'] < 0.6:
                health_status['checks'].append({
                    'name': 'Confidence Check',
                    'status': 'warning',
                    'message': f"Average confidence {self.model_metrics['avg_confidence']:.2f} below threshold 0.6"
                })
                health_status['recommendations'].append("Model may be uncertain - review training data")
            
            # Check 3: Prediction volume
            if self.model_metrics['predictions_per_hour'] < 1:
                health_status['checks'].append({
                    'name': 'Prediction Volume',
                    'status': 'info',
                    'message': f"Low prediction volume: {self.model_metrics['predictions_per_hour']:.1f} per hour"
                })
            
            # Check 4: Error rate
            if self.model_metrics['error_rate'] > 0.3:
                health_status['checks'].append({
                    'name': 'Error Rate',
                    'status': 'critical',
                    'message': f"High error rate: {self.model_metrics['error_rate']:.2f}"
                })
                health_status['recommendations'].append("Immediate model retraining recommended")
                health_status['overall_health'] = 'critical'
            elif self.model_metrics['error_rate'] > 0.2:
                health_status['checks'].append({
                    'name': 'Error Rate',
                    'status': 'warning',
                    'message': f"Elevated error rate: {self.model_metrics['error_rate']:.2f}"
                })
                if health_status['overall_health'] == 'healthy':
                    health_status['overall_health'] = 'warning'
            
            # Add recent alerts
            recent_alerts = [a for a in self.alerts if not a.get('resolved', False)]
            health_status['alerts'] = recent_alerts
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking model health: {e}")
            return {'error': str(e), 'overall_health': 'unknown'}
    
    def _is_recent(self, timestamp: str, hours: int) -> bool:
        """Check if timestamp is within specified hours"""
        try:
            prediction_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return prediction_time > cutoff_time
        except:
            return False
    
    def _check_performance_alerts(self, prediction_data: Dict[str, Any]):
        """Check for performance alerts based on prediction"""
        try:
            # Low confidence alert
            if prediction_data['confidence'] < 0.3:
                alert = {
                    'type': 'low_confidence',
                    'message': f"Very low confidence: {prediction_data['confidence']:.2f}",
                    'timestamp': datetime.now().isoformat(),
                    'prediction': prediction_data['prediction'],
                    'resolved': False
                }
                self.alerts.append(alert)
                logger.warning(f"Low confidence alert: {prediction_data['confidence']:.2f}")
            
            # Check for prediction errors (if actual available)
            if 'actual' in prediction_data and prediction_data['prediction'] != prediction_data['actual']:
                # Check if this is part of a pattern of errors
                recent_errors = [p for p in list(self.predictions_history)[-10:] 
                               if 'actual' in p and p['prediction'] != p['actual']]
                
                if len(recent_errors) >= 5:  # 5+ errors in last 10 predictions
                    alert = {
                        'type': 'high_error_rate',
                        'message': f"High error rate detected: {len(recent_errors)}/10 recent predictions",
                        'timestamp': datetime.now().isoformat(),
                        'resolved': False
                    }
                    self.alerts.append(alert)
                    logger.warning("High error rate alert triggered")
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def export_monitoring_data(self, filepath: str = None) -> Dict[str, Any]:
        """
        Export monitoring data for analysis
        
        Args:
            filepath: Optional file path to save data
            
        Returns:
            Export data dictionary
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'model_metrics': self.get_model_metrics(),
                'performance_trend': self.get_performance_trend(hours=24),
                'health_check': self.check_model_health(),
                'recent_predictions': list(self.predictions_history)[-100:],  # Last 100 predictions
                'alerts': self.alerts[-50:]  # Last 50 alerts
            }
            
            if filepath:
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
                logger.info(f"Monitoring data exported to {filepath}")
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {e}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Create model monitor
    monitor = ModelMonitor()
    
    # Simulate some predictions
    print("🔍 Testing Model Performance Monitoring")
    print("=" * 50)
    
    # Sample predictions with varying confidence and accuracy
    sample_predictions = [
        {'prediction': 'Healthy', 'confidence': 0.95, 'actual': 'Healthy'},
        {'prediction': 'Flu', 'confidence': 0.87, 'actual': 'Flu'},
        {'prediction': 'Digestive', 'confidence': 0.23, 'actual': 'Digestive'},  # Low confidence
        {'prediction': 'Flu', 'confidence': 0.91, 'actual': 'Healthy'},  # Error
        {'prediction': 'Healthy', 'confidence': 0.88, 'actual': 'Healthy'},
        {'prediction': 'Digestive', 'confidence': 0.76, 'actual': 'Digestive'},
        {'prediction': 'Flu', 'confidence': 0.15, 'actual': 'Flu'},  # Very low confidence
        {'prediction': 'Healthy', 'confidence': 0.92, 'actual': 'Flu'},  # Error
        {'prediction': 'Digestive', 'confidence': 0.84, 'actual': 'Digestive'},
        {'prediction': 'Healthy', 'confidence': 0.89, 'actual': 'Healthy'},
    ]
    
    # Log predictions
    for i, pred in enumerate(sample_predictions):
        result = monitor.log_prediction(pred)
        print(f"Prediction {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.2f}) - {result['status']}")
    
    print("\n📊 Model Metrics:")
    metrics = monitor.get_model_metrics()
    print(json.dumps(metrics, indent=2))
    
    print("\n📈 Performance Trend (24h):")
    trend = monitor.get_performance_trend(hours=24)
    print(json.dumps(trend, indent=2))
    
    print("\n🏥 Model Health Check:")
    health = monitor.check_model_health()
    print(json.dumps(health, indent=2))
    
    print("\n✅ Model Performance Monitoring Working Successfully!")
