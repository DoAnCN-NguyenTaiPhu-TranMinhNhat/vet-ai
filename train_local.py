#!/usr/bin/env python3
"""
Standalone training script for local testing
Can be run independently or called from the FastAPI service
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any

# Add the ai_service directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_service'))

from ai_service.training_engine import execute_training, ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 200) -> tuple:
    """Create sample training data for testing"""
    import pandas as pd
    import numpy as np
    
    # Sample feedback data
    animal_types = ['Dog', 'Cat', 'Bird', 'Rabbit']
    genders = ['Male', 'Female']
    seasons = ['Spring', 'Summer', 'Fall', 'Winter']
    vaccinations = ['Up to date', 'Overdue', 'Not vaccinated']
    medical_histories = ['Healthy', 'Chronic condition', 'Previous illness', 'Unknown']
    symptoms_options = [
        'coughing,sneezing,lethargy',
        'vomiting,diarrhea,loss of appetite',
        'limping,swollen paw,difficulty walking',
        'excessive scratching,skin irritation,redness',
        'difficulty breathing,wheezing,coughing'
    ]
    diagnoses = ['Respiratory infection', 'Gastrointestinal issue', 'Musculoskeletal injury', 
                'Skin condition', 'Allergic reaction', 'Healthy']
    
    feedback_data = []
    prediction_logs = []
    
    for i in range(n_samples):
        # Generate random data
        feedback = {
            'prediction_id': i + 1,
            'final_diagnosis': np.random.choice(diagnoses),
            'is_correct': np.random.choice([True, False], p=[0.85, 0.15]),
            'confidence_rating': np.random.randint(3, 6),
            'comments': f'Sample feedback {i+1}',
            'veterinarian_id': np.random.randint(1, 4),
            'is_training_eligible': True,
            'data_quality_score': np.random.uniform(0.7, 1.0),
            'timestamp': datetime.now()
        }
        
        pred_input = {
            'animal_type': np.random.choice(animal_types),
            'gender': np.random.choice(genders),
            'age_months': np.random.randint(1, 180),
            'weight_kg': round(np.random.uniform(1.0, 50.0), 1),
            'temperature': round(np.random.uniform(37.5, 40.5), 1),
            'heart_rate': np.random.randint(60, 180),
            'current_season': np.random.choice(seasons),
            'vaccination_status': np.random.choice(vaccinations),
            'medical_history': np.random.choice(medical_histories),
            'symptom_duration': np.random.randint(1, 30),
            'symptoms_list': np.random.choice(symptoms_options)
        }
        
        prediction = {
            'id': i + 1,
            'visit_id': i + 100,
            'pet_id': i + 50,
            'prediction_input': pred_input,
            'prediction_output': {
                'diagnosis': feedback['final_diagnosis'],
                'confidence': np.random.uniform(0.7, 0.95)
            },
            'model_version': 'v2.0',
            'confidence_score': np.random.uniform(0.7, 0.95),
            'top_k_predictions': [
                {'label': feedback['final_diagnosis'], 'prob': np.random.uniform(0.7, 0.95)}
            ],
            'veterinarian_id': feedback['veterinarian_id'],
            'clinic_id': 1,
            'timestamp': datetime.now()
        }
        
        feedback_data.append(feedback)
        prediction_logs.append(prediction)
    
    logger.info(f"Created {n_samples} sample training records")
    return feedback_data, prediction_logs

def run_local_training(args):
    """Run local training with sample or provided data"""
    logger.info("Starting local training")
    
    if args.use_sample_data:
        # Create sample data
        feedback_data, prediction_logs = create_sample_data(args.n_samples)
    else:
        # Load data from files
        try:
            with open(args.feedback_file, 'r') as f:
                feedback_data = json.load(f)
            with open(args.predictions_file, 'r') as f:
                prediction_logs = json.load(f)
            logger.info(f"Loaded {len(feedback_data)} feedback and {len(prediction_logs)} predictions")
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            return
    
    # Execute training
    result = execute_training(feedback_data, prediction_logs, "local")
    
    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Status: {result['status']}")
    
    if result['status'] == 'completed':
        print(f"Model Version: {result['model_version']}")
        print(f"Model Path: {result['model_path']}")
        
        metrics = result['training_metrics']
        print(f"\nTraining Metrics:")
        print(f"  Training Accuracy: {metrics.get('training_accuracy', 0):.3f}")
        print(f"  Validation Accuracy: {metrics.get('validation_accuracy', 0):.3f}")
        print(f"  Validation F1 Score: {metrics.get('validation_f1', 0):.3f}")
        print(f"  CV Mean Accuracy: {metrics.get('cv_mean_accuracy', 0):.3f}")
        print(f"  Training Time: {metrics.get('training_time_seconds', 0):.1f}s")
        print(f"  Samples: {metrics.get('n_samples', 0)}")
        print(f"  Features: {metrics.get('n_features', 0)}")
        print(f"  Classes: {metrics.get('n_classes', 0)}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("="*60)
    
    # Save results to file
    results_file = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    return result

def test_model_loading(model_version: str):
    """Test loading the trained model"""
    logger.info(f"Testing model loading for version {model_version}")
    
    try:
        import joblib
        from scipy import sparse
        
        model_dir = f"./ai_service/models/{model_version}"
        
        # Load model
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        tab_preprocess = joblib.load(os.path.join(model_dir, "tab_preprocess.pkl"))
        symptoms_mlb = joblib.load(os.path.join(model_dir, "symptoms_mlb.pkl"))
        
        # Load metadata
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        print(f"\nModel loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Classes: {model.classes_}")
        print(f"Training date: {metadata['training_date']}")
        print(f"Training accuracy: {metadata['training_metrics']['training_accuracy']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Vet-AI Training Script')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Run model training')
    train_parser.add_argument('--use-sample-data', action='store_true', 
                            help='Use generated sample data for testing')
    train_parser.add_argument('--n-samples', type=int, default=200,
                            help='Number of sample records to generate')
    train_parser.add_argument('--feedback-file', type=str,
                            help='Path to feedback data JSON file')
    train_parser.add_argument('--predictions-file', type=str,
                            help='Path to predictions data JSON file')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model loading')
    test_parser.add_argument('--model-version', type=str, required=True,
                           help='Model version to test')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_local_training(args)
    elif args.command == 'test':
        test_model_loading(args.model_version)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
