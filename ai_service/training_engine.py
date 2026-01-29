"""
Real ML Training implementation
Handles actual model training with data collection, preprocessing, and model evaluation
"""

import os
import json
import pickle
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import sparse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Real ML model training implementation"""
    
    def __init__(self, model_dir: str = "./ai_service/models", 
                 mlflow_tracking_uri: str = None):
        self.model_dir = model_dir
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        
        # Initialize MLflow
        try:
            # Add timeout to prevent hanging
            import socket
            socket.setdefaulttimeout(5)  # 5 second timeout
            
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("vet-ai-continuous-training")
            self.client = MlflowClient()
            logger.info(f"MLflow tracking initialized at {self.mlflow_tracking_uri}")
            self.mlflow_available = True
        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.client = None
            self.mlflow_available = False
        finally:
            # Reset socket timeout
            import socket
            socket.setdefaulttimeout(None)
        
        # Model parameters
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
    def collect_training_data(self, feedback_data: List[Dict], 
                            prediction_logs: List[Dict]) -> Tuple[pd.DataFrame, pd.Series]:
        """Collect and prepare training data from feedback and predictions"""
        logger.info(f"Collecting training data from {len(feedback_data)} feedback entries")
        
        training_records = []
        
        for feedback in feedback_data:
            # Find corresponding prediction
            prediction_id = feedback['prediction_id']
            
            # Handle both dict and Pydantic object formats
            prediction = None
            for p in prediction_logs:
                if hasattr(p, 'id'):  # Pydantic object
                    if p.id == prediction_id:
                        prediction = p
                        break
                elif isinstance(p, dict):  # Dict object
                    if p.get('id') == prediction_id:
                        prediction = p
                        break
            
            if prediction is None:
                logger.warning(f"No prediction found for feedback {prediction_id}")
                continue
                
            # Extract features from prediction input
            # Handle both dict and Pydantic object formats
            if hasattr(prediction, 'prediction_input'):  # Pydantic object
                pred_input = prediction.prediction_input
            elif isinstance(prediction, dict):  # Dict object
                pred_input = prediction['prediction_input']
            else:
                logger.warning(f"Invalid prediction format for feedback {prediction_id}")
                continue
            
            # Create training record
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
                'final_diagnosis': feedback['final_diagnosis'],
                'data_quality_score': feedback.get('data_quality_score', 1.0)
            }
            
            training_records.append(record)
        
        if not training_records:
            raise ValueError("No valid training records found")
        
        df = pd.DataFrame(training_records)
        
        # Use data quality score as sample weight
        sample_weights = df['data_quality_score'].values
        
        # Remove records with low quality
        df = df[df['data_quality_score'] >= 0.7]
        
        X = df.drop(['final_diagnosis', 'data_quality_score'], axis=1)
        y = df['final_diagnosis']
        
        logger.info(f"Prepared {len(X)} training samples")
        return X, y, sample_weights[:len(X)]
    
    def preprocess_features(self, X: pd.DataFrame, fit_encoders: bool = True) -> Tuple[sparse.csr_matrix, Dict]:
        """Preprocess features with encoders"""
        logger.info("Preprocessing features")
        
        # Separate features
        tabular_features = ['animal_type', 'gender', 'age_months', 'weight_kg', 
                          'temperature', 'heart_rate', 'current_season', 
                          'vaccination_status', 'medical_history', 'symptom_duration']
        
        symptom_feature = 'symptoms_list'
        
        X_tab = X[tabular_features].copy()
        X_sym = X[symptom_feature].copy()
        
        # Process symptoms
        X_sym_processed = X_sym.apply(lambda x: [s.strip().lower() for s in str(x).split(',') if s.strip()])
        
        if fit_encoders:
            # Fit new encoders
            self.tab_preprocess = self._create_tabular_preprocessor()
            X_tab_processed = self.tab_preprocess.fit_transform(X_tab)
            
            self.symptoms_mlb = MultiLabelBinarizer()
            X_sym_encoded = self.symptoms_mlb.fit_transform(X_sym_processed)
        else:
            # Use existing encoders
            if hasattr(self, 'tab_preprocess'):
                X_tab_processed = self.tab_preprocess.transform(X_tab)
            else:
                raise ValueError("No tabular preprocessor found")
                
            if hasattr(self, 'symptoms_mlb'):
                X_sym_encoded = self.symptoms_mlb.transform(X_sym_processed)
            else:
                raise ValueError("No symptoms encoder found")
        
        # Combine features
        X_final = sparse.hstack([X_tab_processed, sparse.csr_matrix(X_sym_encoded)]).tocsr()
        
        # Store preprocessing info
        preprocessing_info = {
            'tabular_features': tabular_features,
            'symptom_feature': symptom_feature,
            'n_symptom_features': X_sym_encoded.shape[1],
            'n_tabular_features': X_tab_processed.shape[1]
        }
        
        logger.info(f"Feature preprocessing complete: {X_final.shape}")
        return X_final, preprocessing_info
    
    def _create_tabular_preprocessor(self):
        """Create tabular feature preprocessor"""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        # Identify categorical and numerical columns
        categorical_cols = ['animal_type', 'gender', 'current_season', 
                          'vaccination_status', 'medical_history']
        numerical_cols = ['age_months', 'weight_kg', 'temperature', 
                         'heart_rate', 'symptom_duration']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ]
        )
        
        return preprocessor
    
    def train_model(self, X: sparse.csr_matrix, y: pd.Series, 
                   sample_weights: np.ndarray = None) -> Dict[str, Any]:
        """Train the actual model"""
        logger.info(f"Starting model training with {X.shape[0]} samples")
        
        # Split data
        X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
            X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(**self.model_params)
        
        start_time = datetime.now()
        model.fit(X_train, y_train, sample_weight=weights_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='accuracy', sample_weight=weights_train)
        
        training_metrics = {
            'training_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy,
            'validation_f1': val_f1,
            'cv_mean_accuracy': cv_scores.mean(),
            'cv_std_accuracy': cv_scores.std(),
            'training_time_seconds': training_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(model.classes_)
        }
        
        logger.info(f"Training completed: val_accuracy={val_accuracy:.3f}, val_f1={val_f1:.3f}")
        
        return model, training_metrics
    
    def save_model(self, model: RandomForestClassifier, 
                   training_metrics: Dict[str, Any], 
                   model_version: str) -> str:
        """Save trained model and artifacts"""
        # Create version directory
        version_dir = os.path.join(self.model_dir, model_version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, "model.pkl")
        joblib.dump(model, model_path)
        
        # Save preprocessors
        if hasattr(self, 'tab_preprocess'):
            joblib.dump(self.tab_preprocess, os.path.join(version_dir, "tab_preprocess.pkl"))
        
        if hasattr(self, 'symptoms_mlb'):
            joblib.dump(self.symptoms_mlb, os.path.join(version_dir, "symptoms_mlb.pkl"))
        
        # Save training metadata
        metadata = {
            'model_version': model_version,
            'training_date': datetime.now().isoformat(),
            'training_metrics': training_metrics,
            'model_params': self.model_params,
            'feature_info': {
                'n_features': training_metrics.get('n_features'),
                'n_classes': training_metrics.get('n_classes'),
                'classes': model.classes_.tolist() if hasattr(model, 'classes_') else []
            }
        }
        
        with open(os.path.join(version_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {version_dir}")
        return version_dir
    
    def log_to_mlflow(self, model: RandomForestClassifier, 
                     training_metrics: Dict[str, Any], 
                     model_version: str,
                     X_val: sparse.csr_matrix = None,
                     y_val: pd.Series = None):
        """Log training to MLflow"""
        if not self.mlflow_available or self.client is None:
            logger.warning("MLflow not available, skipping logging")
            return
        
        try:
            with mlflow.start_run(run_name=f"training-{model_version}") as run:
                # Log parameters
                mlflow.log_params(self.model_params)
                
                # Log metrics
                mlflow.log_metrics(training_metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log artifacts
                if hasattr(self, 'tab_preprocess'):
                    mlflow.sklearn.log_model(self.tab_preprocess, "tab_preprocess")
                
                if hasattr(self, 'symptoms_mlb'):
                    mlflow.sklearn.log_model(self.symptoms_mlb, "symptoms_mlb")
                
                # Log validation dataset info
                if X_val is not None and y_val is not None:
                    mlflow.log_dict({
                        'n_validation_samples': len(y_val),
                        'validation_features': X_val.shape[1]
                    }, "validation_info.json")
                
                logger.info(f"Training logged to MLflow: {run.info.run_id}")
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
    
    def generate_model_version(self) -> str:
        """Generate new model version"""
        try:
            # Get existing versions
            versions = []
            for item in os.listdir(self.model_dir):
                if item.startswith('v') and item[1:].replace('.', '').isdigit():
                    versions.append(item)
            
            if versions:
                # Get latest version
                latest = max(versions, key=lambda x: float(x[1:]))
                major, minor = latest[1:].split('.')
                new_version = f"v{major}.{int(minor) + 1}"
            else:
                new_version = "v1.0"
                
        except Exception:
            new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return new_version

def execute_training(feedback_data: List[Dict], prediction_logs: List[Dict],
                   training_mode: str = "local") -> Dict[str, Any]:
    """Execute training pipeline"""
    logger.info(f"Starting {training_mode} training pipeline")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Collect data
        X, y, sample_weights = trainer.collect_training_data(feedback_data, prediction_logs)
        
        # Preprocess features
        X_processed, preprocessing_info = trainer.preprocess_features(X, fit_encoders=True)
        
        # Train model
        model, training_metrics = trainer.train_model(X_processed, y, sample_weights)
        
        # Generate version
        model_version = trainer.generate_model_version()
        
        # Save model
        model_path = trainer.save_model(model, training_metrics, model_version)
        
        # Log to MLflow
        trainer.log_to_mlflow(model, training_metrics, model_version)
        
        result = {
            'status': 'completed',
            'model_version': model_version,
            'model_path': model_path,
            'training_metrics': training_metrics,
            'preprocessing_info': preprocessing_info,
            'training_mode': training_mode
        }
        
        logger.info(f"Training completed successfully: {model_version}")
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'training_mode': training_mode
        }
