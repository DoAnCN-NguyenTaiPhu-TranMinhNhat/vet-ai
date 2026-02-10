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
    
    def __init__(self, model_dir: str = None, 
                 mlflow_tracking_uri: str = None):
        # Use environment variable MODEL_DIR first, then fallback to default
        if model_dir is None:
            model_dir = os.getenv("MODEL_DIR")
        
        if model_dir is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(script_dir, "models")
        else:
            self.model_dir = os.path.abspath(model_dir)
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Log the model directory for debugging
        logger.info(f"Model directory set to: {self.model_dir}")
        
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "vet-ai-continuous-training")
        
        # Initialize MLflow
        try:
            # Add timeout to prevent hanging
            import socket
            timeout_seconds = int(os.getenv("MLFLOW_TIMEOUT_SECONDS", "5"))
            socket.setdefaulttimeout(timeout_seconds)
            
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment_name)
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
        
    def get_dataset_size_thresholds(self) -> Dict[str, int]:
        """Get configurable dataset size thresholds"""
        return {
            'tiny': int(os.getenv("DATASET_TINY_THRESHOLD", "20")),
            'small': int(os.getenv("DATASET_SMALL_THRESHOLD", "50")),
            'medium': int(os.getenv("DATASET_MEDIUM_THRESHOLD", "200")),
            'large': int(os.getenv("DATASET_LARGE_THRESHOLD", "1000"))
        }
    
    def build_model_params(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Dynamically build model parameters based on dataset characteristics"""
        thresholds = self.get_dataset_size_thresholds()
        n_samples = len(X)
        n_features = X.shape[1] if hasattr(X, 'shape') and len(X.shape) > 1 else len(X.columns)
        n_classes = len(y.unique())
        
        # Adaptive n_estimators: more samples need more trees
        if n_samples < thresholds['small']:
            n_estimators = int(os.getenv("N_ESTIMATORS_SMALL", "50"))
        elif n_samples < thresholds['medium']:
            n_estimators = int(os.getenv("N_ESTIMATORS_MEDIUM", "100"))
        elif n_samples < thresholds['large']:
            n_estimators = int(os.getenv("N_ESTIMATORS_LARGE", "200"))
        else:
            n_estimators = int(os.getenv("N_ESTIMATORS_XLARGE", "300"))
        
        # Adaptive max_depth: prevent overfitting on small datasets
        if n_samples < thresholds['small']:
            max_depth = min(int(os.getenv("MAX_DEPTH_SMALL", "5")), n_features)
        elif n_samples < thresholds['medium']:
            max_depth = min(int(os.getenv("MAX_DEPTH_MEDIUM", "8")), n_features)
        else:
            max_depth = min(int(os.getenv("MAX_DEPTH_LARGE", "15")), n_features)
        
        # Adaptive min_samples_split based on dataset size
        if n_samples < thresholds['small']:
            min_samples_split = int(os.getenv("MIN_SAMPLES_SPLIT_SMALL", "2"))
        elif n_samples < thresholds['medium']:
            min_samples_split = int(os.getenv("MIN_SAMPLES_SPLIT_MEDIUM", "5"))
        else:
            min_samples_split = max(int(os.getenv("MIN_SAMPLES_SPLIT_LARGE", "10")), int(0.02 * n_samples))
        
        # Adaptive min_samples_leaf
        if n_samples < thresholds['small']:
            min_samples_leaf = int(os.getenv("MIN_SAMPLES_LEAF_SMALL", "1"))
        elif n_samples < thresholds['medium']:
            min_samples_leaf = int(os.getenv("MIN_SAMPLES_LEAF_MEDIUM", "2"))
        else:
            min_samples_leaf = max(int(os.getenv("MIN_SAMPLES_LEAF_LARGE", "4")), int(0.01 * n_samples))
        
        return {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'random_state': int(os.getenv("RANDOM_STATE", "42")),
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'class_weight': 'balanced' if n_classes > 2 and n_samples < thresholds['large'] else None
        }
    
    def detect_feature_types(self, X: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically detect numerical and categorical features"""
        categorical_features = []
        numerical_features = []
        
        for column in X.columns:
            # Skip if column has too many missing values
            missing_threshold = float(os.getenv("MISSING_VALUE_THRESHOLD", "0.5"))
            if X[column].isnull().sum() / len(X) > missing_threshold:
                continue
                
            # Detect categorical features
            unique_count = X[column].nunique()
            data_type = X[column].dtype
            
            # Consider categorical if: object type, or few unique values relative to dataset size
            max_unique_values = int(os.getenv("MAX_UNIQUE_VALUES_FOR_CATEGORICAL", "20"))
            unique_ratio_threshold = float(os.getenv("UNIQUE_RATIO_THRESHOLD", "0.05"))
            if (data_type == 'object' or 
                data_type == 'category' or 
                (data_type in ['int64', 'float64'] and unique_count <= min(max_unique_values, len(X) * unique_ratio_threshold))):
                categorical_features.append(column)
            else:
                numerical_features.append(column)
        
        return {
            'categorical': categorical_features,
            'numerical': numerical_features
        }
    
    def compute_quality_threshold(self, quality_scores: np.ndarray) -> float:
        """Compute dynamic quality threshold using percentile"""
        if len(quality_scores) == 0:
            # Use environment variable or default fallback
            fallback_threshold = float(os.getenv("QUALITY_FALLBACK_THRESHOLD", "0.7"))
            return fallback_threshold
        
        # Use configurable percentile and minimum threshold
        percentile = int(os.getenv("QUALITY_PERCENTILE", "25"))
        min_threshold = float(os.getenv("QUALITY_MIN_THRESHOLD", "0.5"))
        threshold = max(min_threshold, np.percentile(quality_scores, percentile))
        return float(threshold)
    
    def get_adaptive_split_ratio(self, n_samples: int) -> float:
        """Get adaptive train/validation split ratio based on dataset size"""
        thresholds = self.get_dataset_size_thresholds()
        
        if n_samples < thresholds['tiny']:
            return float(os.getenv("SPLIT_RATIO_TINY", "0.1"))  # Keep most data for training on tiny datasets
        elif n_samples < thresholds['small']:
            return float(os.getenv("SPLIT_RATIO_SMALL", "0.15"))
        elif n_samples < thresholds['medium']:
            return float(os.getenv("SPLIT_RATIO_MEDIUM", "0.2"))
        else:
            return float(os.getenv("SPLIT_RATIO_LARGE", "0.25"))  # Standard split for larger datasets
    
    def get_adaptive_cv_folds(self, y: pd.Series) -> int:
        """Get adaptive cross-validation folds based on class distribution"""
        class_counts = y.value_counts()
        min_samples_per_class = class_counts.min()
        
        # Ensure at least 2 samples per fold for each class
        max_folds = min_samples_per_class // 2
        max_cv_folds = int(os.getenv("MAX_CV_FOLDS", "5"))
        cv_folds = min(max_cv_folds, max(2, max_folds))
        
        return cv_folds
    
    def collect_training_data(self, feedback_data: List[Dict], 
                            prediction_logs: List[Dict]) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
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
        
        # Compute dynamic quality threshold
        quality_threshold = self.compute_quality_threshold(sample_weights)
        logger.info(f"Using dynamic quality threshold: {quality_threshold:.3f}")
        
        # Remove records with low quality using dynamic threshold
        df = df[df['data_quality_score'] >= quality_threshold]
        
        X = df.drop(['final_diagnosis', 'data_quality_score'], axis=1)
        y = df['final_diagnosis']
        
        logger.info(f"Prepared {len(X)} training samples after quality filtering")
        return X, y, sample_weights[:len(X)]
    
    def preprocess_features(self, X: pd.DataFrame, fit_encoders: bool = True) -> Tuple[sparse.csr_matrix, Dict]:
        """Preprocess features with automatic feature type detection"""
        logger.info("Preprocessing features")
        
        # Automatically detect feature types
        feature_types = self.detect_feature_types(X)
        categorical_features = feature_types['categorical']
        numerical_features = feature_types['numerical']
        
        # Handle symptoms separately if present
        symptom_feature = None
        if 'symptoms_list' in X.columns:
            symptom_feature = 'symptoms_list'
            # Remove symptoms from regular feature processing
            categorical_features = [f for f in categorical_features if f != symptom_feature]
        
        logger.info(f"Detected {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        
        # Prepare feature matrices
        X_features = X.drop(columns=[symptom_feature]) if symptom_feature else X
        
        if fit_encoders:
            # Fit new encoders
            self.tab_preprocess = self._create_tabular_preprocessor(categorical_features, numerical_features)
            X_tab_processed = self.tab_preprocess.fit_transform(X_features)
            
            # Process symptoms if present
            if symptom_feature:
                X_sym = X[symptom_feature].copy()
                X_sym_processed = X_sym.apply(lambda x: [s.strip().lower() for s in str(x).split(',') if s.strip()])
                self.symptoms_mlb = MultiLabelBinarizer()
                X_sym_encoded = self.symptoms_mlb.fit_transform(X_sym_processed)
                X_final = sparse.hstack([X_tab_processed, sparse.csr_matrix(X_sym_encoded)]).tocsr()
            else:
                X_final = X_tab_processed
        else:
            # Use existing encoders
            if hasattr(self, 'tab_preprocess'):
                X_tab_processed = self.tab_preprocess.transform(X_features)
            else:
                raise ValueError("No tabular preprocessor found")
                
            if symptom_feature and hasattr(self, 'symptoms_mlb'):
                X_sym = X[symptom_feature].copy()
                X_sym_processed = X_sym.apply(lambda x: [s.strip().lower() for s in str(x).split(',') if s.strip()])
                X_sym_encoded = self.symptoms_mlb.transform(X_sym_processed)
                X_final = sparse.hstack([X_tab_processed, sparse.csr_matrix(X_sym_encoded)]).tocsr()
            else:
                X_final = X_tab_processed
        
        # Store preprocessing info
        preprocessing_info = {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'symptom_feature': symptom_feature,
            'n_symptom_features': X_sym_encoded.shape[1] if symptom_feature else 0,
            'n_tabular_features': X_tab_processed.shape[1],
            'feature_types': feature_types
        }
        
        logger.info(f"Feature preprocessing complete: {X_final.shape}")
        return X_final, preprocessing_info
    
    def _create_tabular_preprocessor(self, categorical_features: List[str], numerical_features: List[str]):
        """Create tabular feature preprocessor with detected feature types"""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        
        transformers = []
        
        # Add numerical transformer if we have numerical features
        if numerical_features:
            transformers.append(('num', StandardScaler(), numerical_features))
        
        # Add categorical transformer if we have categorical features
        if categorical_features:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
        
        # Create preprocessor with available transformers
        if transformers:
            preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        else:
            # This shouldn't happen with our feature detection, but add safety
            logger.warning("No features detected for preprocessing, using identity transformer")
            preprocessor = ColumnTransformer(
                transformers=[('identity', 'passthrough', [])],
                remainder='passthrough'
            )
        
        return preprocessor
    
    def train_model(self, X: sparse.csr_matrix, y: pd.Series, 
                   sample_weights: np.ndarray = None) -> Dict[str, Any]:
        """Train the actual model with dynamic parameters"""
        logger.info(f"Starting model training with {X.shape[0]} samples")
        
        # Get adaptive split ratio
        n_samples = X.shape[0]  # Use shape[0] instead of len() for sparse matrices
        test_size = self.get_adaptive_split_ratio(n_samples)
        logger.info(f"Using adaptive test split ratio: {test_size}")
        
        # Split data with adaptive ratio
        # Use stratification only if we have enough samples per class
        random_state = int(os.getenv("RANDOM_STATE", "42"))
        
        # Check if stratification is possible
        class_counts = y.value_counts()
        min_samples_per_class = class_counts.min()
        can_stratify = min_samples_per_class >= 2 and len(class_counts) >= 2
        
        if can_stratify:
            try:
                X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
                    X, y, sample_weights, test_size=test_size, random_state=random_state, stratify=y
                )
                logger.info("Used stratified train/test split")
            except ValueError as e:
                if "stratify" in str(e):
                    logger.warning("Cannot stratify due to insufficient samples, using random split")
                    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
                        X, y, sample_weights, test_size=test_size, random_state=random_state
                    )
                else:
                    raise
        else:
            logger.warning(f"Cannot stratify: only {min_samples_per_class} samples in smallest class, using random split")
            X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
                X, y, sample_weights, test_size=test_size, random_state=random_state
            )
        
        # Build dynamic model parameters
        # Use sparse matrix shape for parameter building instead of converting to DataFrame
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        
        # Create a simple DataFrame with just shape info for parameter building
        max_dummy_features = int(os.getenv("MAX_DUMMY_FEATURES", "100"))
        dummy_df = pd.DataFrame(np.random.rand(n_samples, min(n_features, max_dummy_features)))  # Limit columns for efficiency
        model_params = self.build_model_params(dummy_df, y_train)
        logger.info(f"Using dynamic model parameters: {model_params}")
        
        # Train model
        model = RandomForestClassifier(**model_params)
        
        start_time = datetime.now()
        model.fit(X_train, y_train, sample_weight=weights_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        
        # Adaptive cross-validation
        cv_folds = self.get_adaptive_cv_folds(y_train)
        logger.info(f"Using adaptive cross-validation folds: {cv_folds}")
        
        if cv_folds >= 2:
            # Note: fit_params not supported in cross_val_score, sample weights handled during training
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            cv_mean_accuracy = cv_scores.mean()
            cv_std_accuracy = cv_scores.std()
        else:
            # Skip CV if not enough samples per class
            cv_mean_accuracy = train_accuracy
            cv_std_accuracy = 0.0
            logger.warning("Skipping cross-validation due to insufficient samples per class")
        
        training_metrics = {
            'training_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy,
            'validation_f1': val_f1,
            'cv_mean_accuracy': cv_mean_accuracy,
            'cv_std_accuracy': cv_std_accuracy,
            'training_time_seconds': training_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(model.classes_),
            'model_params': model_params,
            'test_split_ratio': test_size,
            'cv_folds': cv_folds
        }
        
        logger.info(f"Training completed: val_accuracy={val_accuracy:.3f}, val_f1={val_f1:.3f}")
        
        return model, training_metrics
    
    def save_model(self, model: RandomForestClassifier, 
                   training_metrics: Dict[str, Any], 
                   model_version: str) -> str:
        """Save trained model and artifacts"""
        try:
            # Create version directory
            version_dir = os.path.join(self.model_dir, model_version)
            os.makedirs(version_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(version_dir, "model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save preprocessors
            if hasattr(self, 'tab_preprocess'):
                tab_path = os.path.join(version_dir, "tab_preprocess.pkl")
                joblib.dump(self.tab_preprocess, tab_path)
                logger.info(f"Tabular preprocessor saved to {tab_path}")
            
            if hasattr(self, 'symptoms_mlb'):
                mlb_path = os.path.join(version_dir, "symptoms_mlb.pkl")
                joblib.dump(self.symptoms_mlb, mlb_path)
                logger.info(f"Symptoms ML encoder saved to {mlb_path}")
            
            # Save training metadata
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return obj.item()
                else:
                    return obj
            
            serializable_metrics = convert_numpy_types(training_metrics)
            
            metadata = {
                'model_version': model_version,
                'training_date': datetime.now().isoformat(),
                'training_metrics': serializable_metrics,
                'model_params': serializable_metrics.get('model_params', {}),
                'feature_info': {
                    'n_features': serializable_metrics.get('n_features'),
                    'n_classes': serializable_metrics.get('n_classes'),
                    'classes': model.classes_.tolist() if hasattr(model, 'classes_') else []
                }
            }
            
            metadata_path = os.path.join(version_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
            
            # Save input schema
            input_schema = {
                'type': 'object',
                'properties': {
                    'age': {'type': 'number'},
                    'weight': {'type': 'number'},
                    'temperature': {'type': 'number'},
                    'symptoms': {
                        'type': 'array',
                        'items': {'type': 'string'}
                    }
                },
                'required': ['age', 'weight', 'temperature', 'symptoms']
            }
            
            schema_path = os.path.join(version_dir, "input_schema.json")
            with open(schema_path, 'w') as f:
                json.dump(input_schema, f, indent=2)
            logger.info(f"Input schema saved to {schema_path}")
            
            # Save metrics separately for easy access
            metrics_path = os.path.join(version_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
            
            logger.info(f"Model successfully saved to {version_dir}")
            
            # Verify all files exist
            expected_files = ["model.pkl", "metadata.json", "input_schema.json", "metrics.json"]
            missing_files = []
            for file_name in expected_files:
                file_path = os.path.join(version_dir, file_name)
                if not os.path.exists(file_path):
                    missing_files.append(file_name)
            
            if missing_files:
                raise Exception(f"Failed to save files: {missing_files}")
            
            return version_dir
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def log_to_mlflow(self, model: RandomForestClassifier, 
                     training_metrics: Dict[str, Any], 
                     model_version: str,
                     X_val: sparse.csr_matrix = None,
                     y_val: pd.Series = None):
        """Log training to MLflow with dynamic experiment name"""
        if not self.mlflow_available or self.client is None:
            logger.warning("MLflow not available, skipping logging")
            return
        
        try:
            with mlflow.start_run(run_name=f"training-{model_version}") as run:
                # Log parameters
                mlflow.log_params(training_metrics.get('model_params', {}))
                
                # Log additional parameters
                default_split_ratio = float(os.getenv("DEFAULT_SPLIT_RATIO", "0.2"))
                default_cv_folds = int(os.getenv("DEFAULT_CV_FOLDS", "5"))
                mlflow.log_param("test_split_ratio", training_metrics.get('test_split_ratio', default_split_ratio))
                mlflow.log_param("cv_folds", training_metrics.get('cv_folds', default_cv_folds))
                mlflow.log_param("experiment_name", self.mlflow_experiment_name)
                
                # Log metrics
                metrics_to_log = {k: v for k, v in training_metrics.items() 
                                 if isinstance(v, (int, float)) and k != 'model_params'}
                mlflow.log_metrics(metrics_to_log)
                
                # Log model with artifact_path instead of name
                mlflow.sklearn.log_model(
                    model, 
                    artifact_path="model",
                    registered_model_name="vet-ai-model"
                )
                
                # Log artifacts using log_artifact instead of log_model for preprocessors
                if hasattr(self, 'tab_preprocess'):
                    joblib.dump(self.tab_preprocess, "tab_preprocess.pkl")
                    mlflow.log_artifact("tab_preprocess.pkl", "preprocessors")
                    os.remove("tab_preprocess.pkl")
                
                if hasattr(self, 'symptoms_mlb'):
                    joblib.dump(self.symptoms_mlb, "symptoms_mlb.pkl")
                    mlflow.log_artifact("symptoms_mlb.pkl", "preprocessors")
                    os.remove("symptoms_mlb.pkl")
                
                # Log dataset info
                dataset_info = {
                    'n_samples': training_metrics.get('n_samples', 0),
                    'n_features': training_metrics.get('n_features', 0),
                    'n_classes': training_metrics.get('n_classes', 0),
                    'training_date': datetime.now().isoformat(),
                    'model_version': model_version
                }
                mlflow.log_dict(dataset_info, "dataset_info.json")
                
                # Log validation dataset info
                if X_val is not None and y_val is not None:
                    validation_info = {
                        'n_validation_samples': len(y_val),
                        'validation_features': X_val.shape[1]
                    }
                    mlflow.log_dict(validation_info, "validation_info.json")
                
                logger.info(f"Training logged to MLflow experiment '{self.mlflow_experiment_name}': {run.info.run_id}")
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {str(e)}")
            # Don't raise - MLflow logging failure should not stop training
    
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
    """Execute training pipeline with dynamic parameters"""
    logger.info(f"Starting {training_mode} training pipeline")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Collect data
        X, y, sample_weights = trainer.collect_training_data(feedback_data, prediction_logs)
        
        # Preprocess features
        X_processed, preprocessing_info = trainer.preprocess_features(X, fit_encoders=True)
        
        # Train model with dynamic parameters
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
            'training_mode': training_mode,
            'dynamic_params': {
                'quality_threshold': training_metrics.get('test_split_ratio'),
                'split_ratio': training_metrics.get('test_split_ratio'),
                'cv_folds': training_metrics.get('cv_folds'),
                'model_params': training_metrics.get('model_params')
            }
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
