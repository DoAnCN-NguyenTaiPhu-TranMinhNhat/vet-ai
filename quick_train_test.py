#!/usr/bin/env python3
"""
Quick test training script without MLflow dependency
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

# Add the ai_service directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai_service'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples: int = 100) -> tuple:
    """Create sample training data for testing"""
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

def quick_train_test():
    """Quick training test without MLflow"""
    logger.info("Starting quick training test")
    
    try:
        # Import minimal training components
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import MultiLabelBinarizer
        from scipy import sparse
        import joblib
        
        # Create sample data
        feedback_data, prediction_logs = create_sample_data(100)
        
        # Prepare training data
        training_records = []
        for feedback in feedback_data:
            prediction_id = feedback['prediction_id']
            prediction = next((p for p in prediction_logs if p.get('id') == prediction_id), None)
            
            if prediction is None:
                continue
                
            pred_input = prediction['prediction_input']
            
            record = {
                'animal_type': pred_input.get('animal_type'),
                'gender': pred_input.get('gender'),
                'age_months': pred_input.get('age_months'),
                'weight_kg': pred_input.get('weight_kg'),
                'temperature': pred_input.get('temperature'),
                'heart_rate': pred_input.get('heart_rate'),
                'current_season': pred_input.get('current_season'),
                'vaccination_status': pred_input.get('vaccination_status'),
                'medical_history': pred_input.get('medical_history', 'Unknown'),
                'symptom_duration': pred_input.get('symptom_duration'),
                'symptoms_list': pred_input.get('symptoms_list', ''),
                'final_diagnosis': feedback['final_diagnosis']
            }
            
            training_records.append(record)
        
        if not training_records:
            logger.error("No valid training records found")
            return {'status': 'failed', 'error': 'No valid training records'}
        
        df = pd.DataFrame(training_records)
        X = df.drop(['final_diagnosis'], axis=1)
        y = df['final_diagnosis']
        
        logger.info(f"Prepared {len(X)} training samples")
        
        # Preprocess features
        tabular_features = ['animal_type', 'gender', 'age_months', 'weight_kg', 
                          'temperature', 'heart_rate', 'current_season', 
                          'vaccination_status', 'medical_history', 'symptom_duration']
        
        symptom_feature = 'symptoms_list'
        
        X_tab = X[tabular_features].copy()
        X_sym = X[symptom_feature].copy()
        
        # Process symptoms
        X_sym_processed = X_sym.apply(lambda x: [s.strip().lower() for s in str(x).split(',') if s.strip()])
        
        # Fit preprocessors
        categorical_cols = ['animal_type', 'gender', 'current_season', 
                          'vaccination_status', 'medical_history']
        numerical_cols = ['age_months', 'weight_kg', 'temperature', 
                         'heart_rate', 'symptom_duration']
        
        tab_preprocess = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        X_tab_processed = tab_preprocess.fit_transform(X_tab)
        
        symptoms_mlb = MultiLabelBinarizer()
        X_sym_encoded = symptoms_mlb.fit_transform(X_sym_processed)
        
        # Combine features
        X_final = sparse.hstack([X_tab_processed, sparse.csr_matrix(X_sym_encoded)]).tocsr()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_final, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        # Save model
        model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        version_dir = f"./ai_service/models/{model_version}"
        os.makedirs(version_dir, exist_ok=True)
        
        joblib.dump(model, os.path.join(version_dir, "model.pkl"))
        joblib.dump(tab_preprocess, os.path.join(version_dir, "tab_preprocess.pkl"))
        joblib.dump(symptoms_mlb, os.path.join(version_dir, "symptoms_mlb.pkl"))
        
        # Save metadata
        metadata = {
            'model_version': model_version,
            'training_date': datetime.now().isoformat(),
            'training_metrics': {
                'training_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy,
                'validation_f1': val_f1,
                'training_time_seconds': training_time,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(model.classes_)
            },
            'model_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'feature_info': {
                'n_features': X.shape[1],
                'n_classes': len(model.classes_),
                'classes': model.classes_.tolist()
            }
        }
        
        with open(os.path.join(version_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        result = {
            'status': 'completed',
            'model_version': model_version,
            'model_path': version_dir,
            'training_metrics': metadata['training_metrics'],
            'training_mode': 'local_quick_test'
        }
        
        logger.info(f"Quick training completed successfully!")
        logger.info(f"Model version: {model_version}")
        logger.info(f"Validation accuracy: {val_accuracy:.3f}")
        logger.info(f"Validation F1: {val_f1:.3f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Quick training failed: {e}")
        return {'status': 'failed', 'error': str(e)}

if __name__ == "__main__":
    result = quick_train_test()
    
    print("\n" + "="*60)
    print("QUICK TRAINING RESULTS")
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
        print(f"  Training Time: {metrics.get('training_time_seconds', 0):.1f}s")
        print(f"  Samples: {metrics.get('n_samples', 0)}")
        print(f"  Features: {metrics.get('n_features', 0)}")
        print(f"  Classes: {metrics.get('n_classes', 0)}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("="*60)
