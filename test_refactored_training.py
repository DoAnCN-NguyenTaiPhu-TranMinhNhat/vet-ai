#!/usr/bin/env python3
"""Test script for refactored training pipeline"""

import pandas as pd
import numpy as np
from ai_service.training_engine import ModelTrainer, execute_training

def create_sample_data(n_samples=50):
    """Create sample data for testing"""
    np.random.seed(42)
    
    feedback_data = []
    prediction_logs = []
    
    for i in range(n_samples):
        # Create feedback
        feedback = {
            'prediction_id': i,
            'final_diagnosis': np.random.choice(['Healthy', 'Sick', 'Critical']),
            'is_correct': np.random.choice([True, False]),
            'data_quality_score': np.random.uniform(0.3, 1.0)
        }
        feedback_data.append(feedback)
        
        # Create prediction
        prediction = {
            'id': i,
            'prediction_input': {
                'animal_type': np.random.choice(['Dog', 'Cat']),
                'gender': np.random.choice(['Male', 'Female']),
                'age_months': np.random.randint(1, 120),
                'weight_kg': np.random.uniform(1, 50),
                'temperature': np.random.uniform(36, 41),
                'heart_rate': np.random.randint(60, 150),
                'current_season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter']),
                'vaccination_status': np.random.choice(['Complete', 'Partial', 'None']),
                'medical_history': np.random.choice(['Clean', 'Previous issues']),
                'symptom_duration': np.random.randint(1, 30),
                'symptoms_list': 'cough, fever, loss of appetite'
            }
        }
        prediction_logs.append(prediction)
    
    return feedback_data, prediction_logs

def test_dynamic_parameters():
    """Test dynamic parameter generation"""
    print("Testing dynamic parameters...")
    
    trainer = ModelTrainer()
    
    # Test with small dataset
    X_small = pd.DataFrame(np.random.rand(20, 5))
    y_small = pd.Series(np.random.choice(['A', 'B'], 20))
    
    params_small = trainer.build_model_params(X_small, y_small)
    print(f"Small dataset params: {params_small}")
    
    # Test with large dataset
    X_large = pd.DataFrame(np.random.rand(500, 10))
    y_large = pd.Series(np.random.choice(['A', 'B', 'C'], 500))
    
    params_large = trainer.build_model_params(X_large, y_large)
    print(f"Large dataset params: {params_large}")
    
    # Test feature detection
    X_mixed = pd.DataFrame({
        'cat_feature': ['A', 'B'] * 10,
        'num_feature': np.random.rand(20),
        'binary_feature': [0, 1] * 10
    })
    
    feature_types = trainer.detect_feature_types(X_mixed)
    print(f"Detected feature types: {feature_types}")
    
    print("Dynamic parameter tests passed!")

def test_full_pipeline():
    """Test full training pipeline"""
    print("\nTesting full training pipeline...")
    
    # Create sample data
    feedback_data, prediction_logs = create_sample_data(30)
    
    # Execute training
    result = execute_training(feedback_data, prediction_logs, "local")
    
    print(f"Training result: {result['status']}")
    if result['status'] == 'completed':
        metrics = result['training_metrics']
        print(f"Validation accuracy: {metrics['validation_accuracy']:.3f}")
        print(f"Dynamic params: {result['dynamic_params']}")
    else:
        print(f"Training failed: {result['error']}")

if __name__ == "__main__":
    test_dynamic_parameters()
    test_full_pipeline()
    print("\nAll tests completed!")
