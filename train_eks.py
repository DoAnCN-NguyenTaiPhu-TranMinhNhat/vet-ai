#!/usr/bin/env python3
"""
EKS Training Script - Runs inside Kubernetes pod on training node group
This script is designed to be executed in a Kubernetes job on EKS
"""

import os
import sys
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data_from_api() -> tuple:
    """Load training data from the FastAPI service"""
    try:
        # Get API endpoint from environment
        api_base_url = os.getenv('API_BASE_URL', 'http://vet-ai-service:8000')
        
        # Fetch feedback data
        feedback_response = requests.get(f"{api_base_url}/continuous-training/feedback/all")
        feedback_response.raise_for_status()
        feedback_data = feedback_response.json()
        
        # Fetch prediction logs
        predictions_response = requests.get(f"{api_base_url}/continuous-training/predictions/all")
        predictions_response.raise_for_status()
        prediction_logs = predictions_response.json()
        
        logger.info(f"Loaded {len(feedback_data)} feedback and {len(prediction_logs)} predictions from API")
        return feedback_data, prediction_logs
        
    except Exception as e:
        logger.error(f"Failed to load data from API: {e}")
        raise

def load_training_data_from_files() -> tuple:
    """Load training data from mounted files (for testing)"""
    try:
        data_dir = os.getenv('TRAINING_DATA_DIR', '/data')
        
        with open(os.path.join(data_dir, 'feedback.json'), 'r') as f:
            feedback_data = json.load(f)
        
        with open(os.path.join(data_dir, 'predictions.json'), 'r') as f:
            prediction_logs = json.load(f)
        
        logger.info(f"Loaded {len(feedback_data)} feedback and {len(prediction_logs)} predictions from files")
        return feedback_data, prediction_logs
        
    except Exception as e:
        logger.error(f"Failed to load data from files: {e}")
        raise

def main():
    """Main training execution function"""
    training_id = os.getenv('TRAINING_ID', 'unknown')
    logger.info(f"Starting EKS training job {training_id}")
    
    try:
        # Add ai_service to path
        sys.path.insert(0, '/app/ai_service')
        from training_engine import execute_training
        
        # Load training data
        data_source = os.getenv('DATA_SOURCE', 'api')  # 'api' or 'files'
        
        if data_source == 'api':
            feedback_data, prediction_logs = load_training_data_from_api()
        else:
            feedback_data, prediction_logs = load_training_data_from_files()
        
        # Validate data
        if not feedback_data or not prediction_logs:
            raise ValueError("No training data available")
        
        # Execute training
        logger.info(f"Executing training with {len(feedback_data)} samples")
        result = execute_training(feedback_data, prediction_logs, "eks_hybrid")
        
        # Output result for log parsing
        result_json = json.dumps(result, default=str)
        print(f"TRAINING_RESULT:{result_json}")
        
        # Log summary
        if result['status'] == 'completed':
            logger.info(f"Training completed successfully!")
            logger.info(f"Model version: {result['model_version']}")
            metrics = result['training_metrics']
            logger.info(f"Validation accuracy: {metrics.get('validation_accuracy', 0):.3f}")
            logger.info(f"Validation F1: {metrics.get('validation_f1', 0):.3f}")
        else:
            logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Training execution failed: {e}")
        error_result = {
            'status': 'failed',
            'error': str(e),
            'training_id': training_id
        }
        print(f"TRAINING_RESULT:{json.dumps(error_result, default=str)}")
        return error_result

if __name__ == "__main__":
    main()
