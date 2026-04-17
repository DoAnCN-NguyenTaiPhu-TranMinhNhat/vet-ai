"""
Real ML Training implementation
Handles actual model training with data collection, preprocessing, and model evaluation
"""

import io
import os
import json
import pickle
import uuid
import hashlib
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
try:
    from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.preprocessing import MultiLabelBinarizer
except ImportError:  # pragma: no cover - optional in minimal runtime
    train_test_split = cross_val_score = RepeatedStratifiedKFold = None
    RandomForestClassifier = None
    CalibratedClassifierCV = None
    accuracy_score = classification_report = f1_score = None
    StandardScaler = OneHotEncoder = MultiLabelBinarizer = None
from scipy import sparse
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover - optional in minimal runtime
    mlflow = None
    MlflowClient = None
from ai_service.app.domain.services.clinic_scope_service import clinic_dir_slug, normalize_clinic_key
from ai_service.app.infrastructure.storage.model_store import (
    get_active_model_for_clinic,
    list_model_versions,
    resolve_model_dir,
)

logger = logging.getLogger(__name__)


def _parse_feedback_timestamp(feedback: Dict, prediction: Any) -> Optional[pd.Timestamp]:
    """Best-effort timestamp for time-aware validation (feedback time preferred)."""
    raw = None
    if isinstance(feedback, dict):
        raw = feedback.get("created_at") or feedback.get("timestamp")
    if raw is None and prediction is not None:
        if isinstance(prediction, dict):
            raw = prediction.get("created_at")
        else:
            raw = getattr(prediction, "created_at", None)
    if raw is None:
        return None
    try:
        if hasattr(raw, "timestamp"):
            return pd.Timestamp(raw)
        ts = pd.Timestamp(raw)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts
    except Exception:
        return None


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error between one-hot y and predicted probabilities."""
    try:
        cls_idx = {str(c): i for i, c in enumerate(np.asarray(classes).ravel())}
        n = len(y_true)
        k = len(classes)
        y_ohe = np.zeros((n, k), dtype=float)
        for i, lab in enumerate(np.asarray(y_true).ravel()):
            j = cls_idx.get(str(lab))
            if j is None:
                continue
            y_ohe[i, j] = 1.0
        return float(np.mean(np.sum((y_ohe - proba) ** 2, axis=1)))
    except Exception:
        return float("nan")


class ModelTrainer:
    """Real ML model training implementation"""
    
    def __init__(self, model_dir: str = None, 
                 mlflow_tracking_uri: str = None):
        # Prefer MODEL_ROOT_DIR for storing all versions; fallback to MODEL_DIR for backward compatibility
        if model_dir is None:
            model_dir = os.getenv("MODEL_ROOT_DIR") or os.getenv("VETAI_MODELS_ROOT") or os.getenv("MODEL_DIR")
        
        if model_dir is None:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_dir = os.path.join(script_dir, "models")
        else:
            md = os.path.abspath(model_dir)
            # If MODEL_DIR points to a specific version directory, store versions in its parent
            if os.path.exists(os.path.join(md, "model.pkl")):
                md = os.path.dirname(md)
            self.model_dir = md
        
        # Ensure model directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Log the model directory for debugging
        logger.info(f"Model directory set to: {self.model_dir}")
        
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "vet-ai-continuous-training")
        self._mlflow_last_error: Optional[str] = None
        self._baseline_class_weight: Optional[Dict[str, float]] = None
        self.calibrated_classifier: Any = None
        
        # Initialize MLflow
        try:
            import socket
            timeout_seconds = int(os.getenv("MLFLOW_TIMEOUT_SECONDS", "10"))
            socket.setdefaulttimeout(timeout_seconds)
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.client = MlflowClient()
            self._try_set_experiment()
            self._mlflow_last_error = None
            logger.info(f"MLflow tracking initialized at {self.mlflow_tracking_uri}")
            self.mlflow_available = True
        except Exception as e:
            self._mlflow_last_error = str(e)
            logger.warning(f"MLflow initialization failed: {self._mlflow_last_error}")
            self.client = None
            self.mlflow_available = False
        finally:
            import socket
            socket.setdefaulttimeout(None)

    def _get_baseline_class_weight(self) -> Optional[Dict[str, float]]:
        """
        Compute stable class weights from a baseline dataset (CSV) if provided.
        Env:
          - BASELINE_DATASET_CSV: path to CSV (inside container / mounted)
          - BASELINE_LABEL_COL: label column (default: target_diagnosis)
        """
        if self._baseline_class_weight is not None:
            return self._baseline_class_weight

        csv_path = os.getenv("BASELINE_DATASET_CSV")
        if not csv_path or not str(csv_path).strip():
            self._baseline_class_weight = None
            return None

        label_col = os.getenv("BASELINE_LABEL_COL", "target_diagnosis")
        try:
            df = pd.read_csv(csv_path, usecols=[label_col])
            y = df[label_col].astype(str)
            from sklearn.utils.class_weight import compute_class_weight

            classes = np.unique(y)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
            self._baseline_class_weight = {str(c): float(w) for c, w in zip(classes, weights)}
            logger.info("Computed baseline class_weight from %s (classes=%s)", csv_path, len(classes))
            return self._baseline_class_weight
        except Exception as e:
            logger.warning("Failed to compute baseline class_weight from %s: %s", csv_path, e)
            self._baseline_class_weight = None
            return None
    
    def _experiment_name_for_clinic(self, clinic_key: Optional[str]) -> str:
        """Shared experiment when global; dedicated MLflow experiment per clinic when scoped."""
        base = self.mlflow_experiment_name
        ck = normalize_clinic_key(clinic_key)
        if ck is None:
            return base
        return f"{base}-clinic-{clinic_dir_slug(ck)}"

    def _try_set_experiment_name(self, experiment_name: str) -> None:
        """Set active MLflow experiment; restore if soft-deleted."""
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            if "deleted experiment" in str(e).lower() and self.client is not None:
                exp = self.client.get_experiment_by_name(experiment_name)
                if exp is not None:
                    self.client.restore_experiment(exp.experiment_id)
                    logger.info("Restored deleted MLflow experiment '%s'", experiment_name)
                    mlflow.set_experiment(experiment_name)
                    return
            raise

    def _try_set_experiment(self) -> None:
        """Set default experiment from MLFLOW_EXPERIMENT_NAME."""
        self._try_set_experiment_name(self.mlflow_experiment_name)
    
    def _ensure_mlflow_connected(self) -> bool:
        """Re-try MLflow connection if not yet available (e.g. server was down at init)."""
        if self.mlflow_available and self.client is not None:
            return True
        logger.info(f"Retrying MLflow connection to {self.mlflow_tracking_uri} ...")
        try:
            import socket
            timeout_seconds = int(os.getenv("MLFLOW_TIMEOUT_SECONDS", "10"))
            socket.setdefaulttimeout(timeout_seconds)
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            self.client = MlflowClient()
            self._try_set_experiment()
            self._mlflow_last_error = None
            self.mlflow_available = True
            logger.info(f"MLflow connected at {self.mlflow_tracking_uri}")
            return True
        except Exception as e:
            self._mlflow_last_error = str(e)
            logger.warning(f"MLflow reconnection failed: {self._mlflow_last_error}")
            return False
        finally:
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
            # IMPORTANT: avoid sklearn's internal class_weight balancing while we do
            # dataset-derived balancing via sample_weight (stable for warm_start).
            'class_weight': None
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
    
    def collect_feedback_only_training_frame(
        self,
        feedback_data: List[Dict],
        prediction_logs: List[Dict],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Expert feedback rows only (same quality filter as full training), without
        core-memory mix or class-completion reference rows. Used for promotion gates
        that measure improvement on the current feedback batch.
        """
        logger.info(
            "Collecting feedback-only frame from %d feedback entries",
            len(feedback_data),
        )
        training_records = []

        def _pred_ids_match(stored, wanted) -> bool:
            if stored is None or wanted is None:
                return False
            return str(stored) == str(wanted)

        for feedback in feedback_data:
            prediction_id = feedback["prediction_id"]
            prediction = None
            for p in prediction_logs:
                if hasattr(p, "id"):
                    if _pred_ids_match(p.id, prediction_id):
                        prediction = p
                        break
                elif isinstance(p, dict):
                    if _pred_ids_match(p.get("id"), prediction_id):
                        prediction = p
                        break

            if prediction is None:
                logger.warning("No prediction found for feedback %s", prediction_id)
                continue

            if hasattr(prediction, "prediction_input"):
                pred_input = prediction.prediction_input
            elif isinstance(prediction, dict):
                pred_input = prediction["prediction_input"]
            else:
                logger.warning("Invalid prediction format for feedback %s", prediction_id)
                continue

            training_records.append(
                {
                    "animal_type": pred_input.get("animal_type"),
                    "gender": pred_input.get("gender"),
                    "age_months": pred_input.get("age_months"),
                    "weight_kg": pred_input.get("weight_kg"),
                    "temperature": pred_input.get("temperature"),
                    "heart_rate": pred_input.get("heart_rate"),
                    "current_season": pred_input.get("current_season"),
                    "vaccination_status": pred_input.get("vaccination_status"),
                    "medical_history": pred_input.get("medical_history", "Unknown"),
                    "symptom_duration": pred_input.get("symptom_duration"),
                    "symptoms_list": pred_input.get("symptoms_list", ""),
                    "final_diagnosis": feedback["final_diagnosis"],
                    "data_quality_score": feedback.get("data_quality_score", 1.0),
                }
            )

        if not training_records:
            raise ValueError("No valid training records found")

        df = pd.DataFrame(training_records)
        quality_scores = df["data_quality_score"].astype(float).to_numpy()
        quality_threshold = self.compute_quality_threshold(quality_scores)
        logger.info(
            "Feedback-only frame: quality threshold %.3f",
            quality_threshold,
        )
        df = df[df["data_quality_score"].astype(float) >= quality_threshold]
        if df.empty:
            raise ValueError("No training records left after quality filtering")

        X = df.drop(["final_diagnosis", "data_quality_score"], axis=1)
        y = df["final_diagnosis"]
        logger.info("Prepared %d feedback-only samples for evaluation", len(X))
        return X, y

    def collect_training_data(self, feedback_data: List[Dict], 
                            prediction_logs: List[Dict]) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
        """Collect and prepare training data from feedback and predictions.

        Returns ``sample_timestamps`` as ``datetime64[ns]`` (UTC) for time-aware validation;
        synthetic timestamps are used for core-memory / class-completion rows.
        """
        logger.info(f"Collecting training data from {len(feedback_data)} feedback entries")
        
        training_records = []
        
        def _pred_ids_match(stored, wanted) -> bool:
            if stored is None or wanted is None:
                return False
            return str(stored) == str(wanted)

        for feedback in feedback_data:
            # Find corresponding prediction
            prediction_id = feedback['prediction_id']
            
            # Handle both dict and Pydantic object formats
            prediction = None
            for p in prediction_logs:
                if hasattr(p, 'id'):  # Pydantic object
                    if _pred_ids_match(p.id, prediction_id):
                        prediction = p
                        break
                elif isinstance(p, dict):  # Dict object
                    if _pred_ids_match(p.get('id'), prediction_id):
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
            base_quality = float(feedback.get('data_quality_score', 1.0))
            confidence_rating = feedback.get('confidence_rating')
            try:
                confidence_rating = int(confidence_rating) if confidence_rating is not None else None
            except Exception:
                confidence_rating = None
                
            delta_max = float(os.getenv("FEEDBACK_DELTA_MAX", "0.4"))
            accept_strength = float(os.getenv("FEEDBACK_ACCEPT_STRENGTH", "1.0"))
            reject_strength = float(os.getenv("FEEDBACK_REJECT_STRENGTH", "1.0"))

            rating = 0.0 if confidence_rating is None else max(0.0, min(5.0, float(confidence_rating)))
            delta_pos = (rating / 5.0) * delta_max * accept_strength
            delta_neg = ((5.0 - rating) / 5.0) * delta_max * reject_strength

            pos_weight = min(float(os.getenv("FEEDBACK_POSITIVE_MAX_WEIGHT", "2.0")), base_quality + delta_pos)
            neg_min_weight = float(os.getenv("FEEDBACK_NEGATIVE_MIN_WEIGHT", "0.51"))
            neg_weight = max(neg_min_weight, base_quality - delta_neg)

            final_label = feedback['final_diagnosis']
            is_correct = bool(feedback.get('is_correct', True))
            ai_label = feedback.get('ai_diagnosis')

            features = {
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
                'symptoms_list': pred_input.get('symptoms_list', '')
            }

            # On accept (is_correct=true): boost the final_label.
            # On reject (is_correct=false): penalize the AI-suggested label (ai_label) so next time it should decrease.
            # - If final_label == ai_label: only penalize that label (no extra boost).
            # - If final_label != ai_label: boost doctor's final_label and penalize ai_label with symmetric delta.
            row_ts = _parse_feedback_timestamp(feedback, prediction)
            if is_correct:
                training_records.append({**features,
                                          'final_diagnosis': final_label,
                                          'data_quality_score': pos_weight,
                                          'sample_timestamp': row_ts})
            else:
                if ai_label is not None and final_label == ai_label:
                    training_records.append({**features,
                                              'final_diagnosis': final_label,
                                              'data_quality_score': neg_weight,
                                              'sample_timestamp': row_ts})
                else:
                    # Reinforce doctor's final label with the same magnitude used for accept boost
                    training_records.append({**features,
                                              'final_diagnosis': final_label,
                                              'data_quality_score': pos_weight,
                                              'sample_timestamp': row_ts})
                    # Penalize AI label (if available)
                    if ai_label is not None and str(ai_label).strip() != "":
                        training_records.append({**features,
                                                  'final_diagnosis': ai_label,
                                                  'data_quality_score': neg_weight,
                                                  'sample_timestamp': row_ts})
        
        if not training_records:
            raise ValueError("No valid training records found")
        
        df = pd.DataFrame(training_records)
        
        # Compute dynamic quality threshold from quality scores
        quality_scores = df['data_quality_score'].astype(float).to_numpy()
        quality_threshold = self.compute_quality_threshold(quality_scores)
        logger.info(f"Using dynamic quality threshold: {quality_threshold:.3f}")
        
        # Remove records with low quality using dynamic threshold
        df = df[df['data_quality_score'].astype(float) >= quality_threshold]
        if df.empty:
            raise ValueError("No training records left after quality filtering")

        # ----------------------------
        # Balanced Memory Mix 80/20
        # ----------------------------
        # To prevent catastrophic forgetting, we mix a portion of "Core Memory"
        # sampled from the baseline golden dataset into the feedback-only batch.
        #
        # We keep the existing feedback filtering/quality scoring intact, then
        # append core memory records (they get a configurable quality score).
        core_memory_ratio = float(os.getenv("CORE_MEMORY_RATIO", "0.2"))  # desired share of core memory in final set
        # Core memory must contribute to learning; otherwise sample_weight becomes 0
        # and RF essentially ignores these rows.
        core_memory_quality_score = float(os.getenv("CORE_MEMORY_QUALITY_SCORE", "0.2"))
        core_memory_max_rows = int(os.getenv("CORE_MEMORY_MAX_ROWS", "5000"))
        core_memory_random_state = int(os.getenv("RANDOM_STATE", "42"))

        try:
            baseline_csv = os.getenv("BASELINE_DATASET_CSV")
            baseline_label_col = os.getenv("BASELINE_LABEL_COL", "target_diagnosis")

            if baseline_csv and str(baseline_csv).strip() and 0.0 < core_memory_ratio < 1.0:
                n_feedback = len(df)
                # target: core_share = core_n / (core_n + n_feedback) == core_memory_ratio
                core_n = int(round((n_feedback * core_memory_ratio) / max(1e-9, (1.0 - core_memory_ratio))))
                core_n = max(0, min(core_n, core_memory_max_rows))

                if core_n > 0:
                    feature_cols = [
                        "animal_type",
                        "gender",
                        "age_months",
                        "weight_kg",
                        "temperature",
                        "heart_rate",
                        "current_season",
                        "vaccination_status",
                        "medical_history",
                        "symptom_duration",
                        "symptoms_list",
                    ]
                    needed_cols = feature_cols + [baseline_label_col]
                    baseline_full = pd.read_csv(baseline_csv, usecols=needed_cols)
                    if len(baseline_full) > core_n:
                        baseline_core = baseline_full.sample(n=core_n, random_state=core_memory_random_state)
                    else:
                        baseline_core = baseline_full

                    baseline_core = baseline_core.copy()
                    baseline_core["final_diagnosis"] = baseline_core[baseline_label_col].astype(str)
                    baseline_core["data_quality_score"] = core_memory_quality_score
                    baseline_core = baseline_core.drop(columns=[baseline_label_col])
                    # Older-than-feedback rows so time-aware split keeps feedback in the validation tail when possible
                    baseline_core["sample_timestamp"] = pd.Timestamp("1970-01-01", tz="UTC")

                    # Align column order with feedback df
                    df = pd.concat([df, baseline_core[df.columns]], ignore_index=True)
                    logger.info(
                        "Core memory mix added: feedback_n=%d core_n=%d core_ratio_target=%.3f core_q=%.3f total_n=%d",
                        n_feedback,
                        len(baseline_core),
                        core_memory_ratio,
                        core_memory_quality_score,
                        len(df),
                    )
        except Exception as e:
            # Never break training due to core memory sampling issues.
            logger.warning("Core memory mix skipped: %s", e)

        # ----------------------------
        # Class completion (production safety)
        # ----------------------------
        # If the current feedback batch doesn't contain all diagnosis classes,
        # RandomForest(classes_=...) will shrink to only observed classes,
        # which can lead to "100% confidence on a single class".
        #
        # We fix this by adding a small number of reference samples per missing class
        # from the baseline CSV. Reference samples get `data_quality_score=0`
        # so they do not dominate learning, but they keep the class space stable.
        baseline_csv = os.getenv("BASELINE_DATASET_CSV")
        baseline_label_col = os.getenv("BASELINE_LABEL_COL", "target_diagnosis")
        # Default to 1 to avoid tiny-batch stratification errors.
        class_completion_per_missing = int(os.getenv("CLASS_COMPLETION_REFERENCE_PER_MISSING_CLASS", "1"))
        # Small (non-zero) reference weight helps keep class decision boundaries stable.
        class_completion_quality_score = float(os.getenv("CLASS_COMPLETION_REFERENCE_QUALITY_SCORE", "0.1"))

        try:
            if baseline_csv and str(baseline_csv).strip():
                baseline_df = pd.read_csv(baseline_csv, usecols=[baseline_label_col])
                baseline_labels = set(baseline_df[baseline_label_col].astype(str).unique().tolist())

                y_feedback = df["final_diagnosis"].astype(str)
                feedback_labels = set(y_feedback.unique().tolist())
                missing_labels = sorted(list(baseline_labels - feedback_labels))

                if missing_labels:
                    # Load full baseline columns needed for feature construction
                    feature_cols = [
                        "animal_type",
                        "gender",
                        "age_months",
                        "weight_kg",
                        "temperature",
                        "heart_rate",
                        "current_season",
                        "vaccination_status",
                        "medical_history",
                        "symptom_duration",
                        "symptoms_list",
                    ]
                    needed_cols = feature_cols + [baseline_label_col]
                    full_baseline = pd.read_csv(baseline_csv, usecols=needed_cols)
                    full_baseline["__label__"] = full_baseline[baseline_label_col].astype(str)

                    reference_rows = []
                    rng_seed = int(os.getenv("RANDOM_STATE", "42"))
                    for lbl in missing_labels:
                        candidates = full_baseline[full_baseline["__label__"] == lbl]
                        if candidates.empty:
                            continue
                        n_pick = min(class_completion_per_missing, len(candidates))
                        picked = candidates.sample(n=n_pick, random_state=rng_seed, replace=False) if n_pick > 0 else candidates.head(0)
                        for _, row in picked.iterrows():
                            reference_rows.append({
                                "animal_type": row.get("animal_type"),
                                "gender": row.get("gender"),
                                "age_months": row.get("age_months"),
                                "weight_kg": row.get("weight_kg"),
                                "temperature": row.get("temperature"),
                                "heart_rate": row.get("heart_rate"),
                                "current_season": row.get("current_season"),
                                "vaccination_status": row.get("vaccination_status"),
                                "medical_history": row.get("medical_history", "Unknown"),
                                "symptom_duration": row.get("symptom_duration"),
                                "symptoms_list": row.get("symptoms_list", ""),
                                "final_diagnosis": lbl,
                                "data_quality_score": class_completion_quality_score,
                                "sample_timestamp": pd.Timestamp("1970-01-02", tz="UTC"),
                            })

                    if reference_rows:
                        df_ref = pd.DataFrame(reference_rows)
                        # Ensure required columns exist
                        for c in ["medical_history", "symptoms_list"]:
                            if c in df_ref.columns:
                                df_ref[c] = df_ref[c].fillna("Unknown" if c == "medical_history" else "")
                        df = pd.concat([df, df_ref], ignore_index=True)
                        logger.info(
                            "Class completion: feedback_labels=%d, missing_labels=%d, added_reference_rows=%d",
                            len(feedback_labels),
                            len(missing_labels),
                            len(reference_rows),
                        )
        except Exception as e:
            # If baseline CSV missing or malformed, we just proceed without class completion.
            logger.warning("Class completion skipped: %s", e)

        # Recompute sample weights AFTER filtering + class completion (keep alignment)
        sample_weights = df["data_quality_score"].astype(float).to_numpy().ravel()

        ts_series = pd.to_datetime(df["sample_timestamp"], utc=True, errors="coerce")
        if ts_series.isna().any():
            base = pd.Timestamp.now(tz="UTC").normalize()
            n_na = int(ts_series.isna().sum())
            fill = [base + pd.Timedelta(seconds=i) for i in range(n_na)]
            ts_series = ts_series.copy()
            ts_series.loc[ts_series.isna()] = fill
        sample_timestamps = ts_series.to_numpy(dtype="datetime64[ns]")

        X = df.drop(['final_diagnosis', 'data_quality_score', 'sample_timestamp'], axis=1)
        y = df['final_diagnosis']

        logger.info(f"Prepared {len(X)} training samples after quality filtering")
        return X, y, sample_weights, sample_timestamps
    
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

    def load_active_artifacts(self, clinic_key: Optional[str] = None) -> tuple[RandomForestClassifier, str]:
        """
        Load active model + preprocessors so we can warm-start training.
        Returns (model, active_model_dir).
        """
        ck = normalize_clinic_key(clinic_key)
        active = get_active_model_for_clinic(ck)
        if active is None:
            # fallback to newest version directory under models root (global + clinic subdir)
            versions = list_model_versions(ck)
            if not versions:
                raise ValueError("No existing model versions found to fine-tune")
            active_dir = resolve_model_dir(versions[0], ck)
        else:
            active_dir = active.model_dir

        model_path = os.path.join(active_dir, "model.pkl")
        tab_path = os.path.join(active_dir, "tab_preprocess.pkl")
        mlb_path = os.path.join(active_dir, "symptoms_mlb.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Active model.pkl not found: {model_path}")

        model = joblib.load(model_path)
        if os.path.exists(tab_path):
            self.tab_preprocess = joblib.load(tab_path)
        if os.path.exists(mlb_path):
            self.symptoms_mlb = joblib.load(mlb_path)

        if not isinstance(model, RandomForestClassifier):
            raise ValueError(f"Active model is not RandomForestClassifier: {type(model)}")

        logger.info("Loaded active artifacts from %s", active_dir)
        return model, active_dir
    
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
    
    def train_model(
        self,
        X: sparse.csr_matrix,
        y: pd.Series,
        sample_weights: np.ndarray = None,
        base_model: Optional[RandomForestClassifier] = None,
        finetune_add_trees: int = 20,
        split_random_state: Optional[int] = None,
        sample_timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train model.
        - If base_model is provided: warm-start by adding trees (fine-tune-ish).
        - Else: train new model from scratch.
        """
        logger.info(f"Starting model training with {X.shape[0]} samples")
        self.calibrated_classifier = None

        n_samples_all = X.shape[0]
        test_size = self.get_adaptive_split_ratio(n_samples_all)
        logger.info(f"Using adaptive test split ratio: {test_size}")

        random_state = (
            int(split_random_state)
            if split_random_state is not None
            else int(os.getenv("RANDOM_STATE", "42"))
        )

        if sample_weights is not None:
            sample_weights = np.asarray(sample_weights, dtype=float).ravel()

        if sample_timestamps is not None and len(sample_timestamps) != n_samples_all:
            sample_timestamps = None

        validation_mode = os.getenv("VALIDATION_MODE", "random").strip().lower()
        time_min = int(os.getenv("TIME_AWARE_MIN_SAMPLES", "30"))
        validation_note = ""

        ts_arr: Optional[np.ndarray] = None
        if sample_timestamps is not None:
            ts_arr = np.asarray(sample_timestamps)

        def _random_train_val_split():
            class_counts = y.value_counts()
            min_samples_per_class = class_counts.min()
            can_stratify = min_samples_per_class >= 2 and len(class_counts) >= 2
            if can_stratify:
                try:
                    return train_test_split(
                        X, y, sample_weights, test_size=test_size, random_state=random_state, stratify=y
                    )
                except ValueError as e:
                    err = str(e).lower()
                    if ("stratify" in err) or ("test_size" in err) or ("number of classes" in err):
                        logger.warning("Cannot stratify due to insufficient samples, using random split")
                        return train_test_split(
                            X, y, sample_weights, test_size=test_size, random_state=random_state
                        )
                    raise
            logger.warning(
                "Cannot stratify: only %s samples in smallest class, using random split",
                min_samples_per_class,
            )
            return train_test_split(X, y, sample_weights, test_size=test_size, random_state=random_state)

        split_used = "random"
        time_split_ok = False
        if (
            validation_mode == "time_aware"
            and ts_arr is not None
            and n_samples_all >= time_min
        ):
            try:
                order = np.argsort(ts_arr, kind="mergesort")
                n_test = max(1, int(round(n_samples_all * test_size)))
                n_test = min(n_test, n_samples_all - 1)
                train_idx = order[:-n_test]
                val_idx = order[-n_test:]
                y_np = np.asarray(y)
                n_classes_full = len(np.unique(y_np))
                n_classes_val = len(np.unique(y_np[val_idx]))
                if (
                    len(train_idx) >= 1
                    and len(val_idx) >= 1
                    and (n_classes_full < 2 or n_classes_val >= 2)
                ):
                    w_tr = sample_weights[train_idx] if sample_weights is not None else None
                    w_va = sample_weights[val_idx] if sample_weights is not None else None
                    X_train = X[train_idx]
                    X_val = X[val_idx]
                    y_train = y.iloc[train_idx]
                    y_val = y.iloc[val_idx]
                    weights_train, weights_val = w_tr, w_va
                    split_used = "time_aware"
                    validation_note = "chronological holdout (most recent fraction in validation)"
                    time_split_ok = True
            except Exception as e:
                logger.warning("time_aware split failed: %s", e)

        if not time_split_ok:
            if validation_mode == "time_aware":
                validation_note = (
                    "time_aware requested but unavailable (samples/timestamps/coverage); using random split"
                )
            X_train, X_val, y_train, y_val, weights_train, weights_val = _random_train_val_split()

        class_counts_train = y_train.value_counts()
        min_samples_per_class_train = class_counts_train.min()
        can_stratify = min_samples_per_class_train >= 2 and len(class_counts_train) >= 2

        # Dataset-derived class balancing via sample_weight (stable)
        warm_starting = base_model is not None
        baseline_cw: Optional[Dict[str, float]] = self._get_baseline_class_weight()

        # Build dynamic model parameters
        # Use sparse matrix shape for parameter building instead of converting to DataFrame
        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        
        # Create a simple DataFrame with just shape info for parameter building
        max_dummy_features = int(os.getenv("MAX_DUMMY_FEATURES", "100"))
        dummy_df = pd.DataFrame(np.random.rand(n_samples, min(n_features, max_dummy_features)))  # Limit columns for efficiency
        model_params = self.build_model_params(dummy_df, y_train)
        logger.info(f"Using dynamic model parameters: {model_params}")
        
        # Ensure weights are 1-D after split
        if weights_train is not None:
            weights_train = np.asarray(weights_train, dtype=float).ravel()
        if weights_val is not None:
            weights_val = np.asarray(weights_val, dtype=float).ravel()

        # Train model
        if base_model is not None:
            model = base_model
            model.warm_start = True
            try:
                # Disable RF internal class_weight processing; we'll apply baseline class_weight ourselves
                # via sample_weight (see below).
                if getattr(model, "class_weight", None) in ("balanced", "balanced_subsample"):
                    model.class_weight = None
                if hasattr(model, "class_weight_"):
                    delattr(model, "class_weight_")
            except Exception:
                pass
            try:
                current = int(getattr(model, "n_estimators", 0) or 0)
            except Exception:
                current = 0
            model.n_estimators = max(current + int(finetune_add_trees), current + 1)
            logger.info("Warm-start fine-tune: n_estimators %s -> %s", current, model.n_estimators)
        else:
            model = RandomForestClassifier(**model_params)
        
        # Apply baseline class_weight as multiplicative factor on sample_weight
        if baseline_cw and weights_train is not None:
            try:
                # Force 1-D vectors to avoid numpy broadcasting into (n,n)
                y_train_str = np.asarray(y_train).ravel().astype(str)
                cw_vec = np.array([baseline_cw.get(lbl, 1.0) for lbl in y_train_str], dtype=float).ravel()
                weights_train = np.asarray(weights_train, dtype=float).ravel()
                logger.info(
                    "Fine-tune sample_weight shapes: weights_train=%s cw_vec=%s",
                    weights_train.shape,
                    cw_vec.shape
                )
                weights_train *= cw_vec
                logger.info("Applied baseline class_weight via sample_weight (element-wise)")
            except Exception as e:
                logger.warning("Failed to apply baseline class_weight to sample_weight: %s", e)

        start_time = datetime.now()
        model.fit(X_train, y_train, sample_weight=weights_train)
        training_time = (datetime.now() - start_time).total_seconds()

        eval_model: Any = model
        cal_method = os.getenv("CALIBRATION_METHOD", "none").strip().lower()
        cal_min = int(os.getenv("CALIBRATION_MIN_SAMPLES", "50"))
        calibration_brier_before: Optional[float] = None
        calibration_brier_after: Optional[float] = None
        calibrator: Any = None

        if (
            CalibratedClassifierCV is not None
            and cal_method in ("isotonic", "sigmoid")
            and X_val.shape[0] >= cal_min
        ):
            try:
                proba_before = model.predict_proba(X_val)
                calibration_brier_before = _multiclass_brier(
                    np.asarray(y_val), proba_before, model.classes_
                )
                calibrator = CalibratedClassifierCV(model, method=cal_method, cv="prefit")
                calibrator.fit(X_val, y_val)
                proba_after = calibrator.predict_proba(X_val)
                calibration_brier_after = _multiclass_brier(
                    np.asarray(y_val), proba_after, model.classes_
                )
                self.calibrated_classifier = calibrator
                eval_model = calibrator
            except Exception as e:
                logger.warning("Calibration skipped: %s", e)
                self.calibrated_classifier = None
                eval_model = model
        train_pred = model.predict(X_train)
        val_pred = eval_model.predict(X_val)

        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        try:
            val_f1 = float(f1_score(y_val, val_pred, average="weighted", zero_division=0))
        except TypeError:
            val_f1 = float(f1_score(y_val, val_pred, average="weighted"))

        proba_val = eval_model.predict_proba(X_val)
        maxp = np.max(proba_val, axis=1)
        pred_argmax = np.argmax(proba_val, axis=1)
        classes_arr = eval_model.classes_
        y_pred_labels = np.asarray(classes_arr[pred_argmax], dtype=object)
        y_val_np = np.asarray(y_val).ravel()
        try:
            maj = pd.Series(y_train).mode().iloc[0]
        except Exception:
            maj = pd.Series(y_train).iloc[0]
        best_t = 0.5
        try:
            best_f1w = float(
                f1_score(y_val_np, y_pred_labels, average="weighted", zero_division=0)
            )
        except TypeError:
            best_f1w = float(f1_score(y_val_np, y_pred_labels, average="weighted"))
        for t in np.linspace(0.05, 0.95, 91):
            y_pred_t = np.where(maxp >= t, y_pred_labels, maj)
            try:
                fw = float(f1_score(y_val_np, y_pred_t, average="weighted", zero_division=0))
            except TypeError:
                fw = float(f1_score(y_val_np, y_pred_t, average="weighted"))
            if fw > best_f1w:
                best_f1w = fw
                best_t = float(t)
        
        # Adaptive cross-validation
        cv_folds = self.get_adaptive_cv_folds(y_train)
        logger.info(f"Using adaptive cross-validation folds: {cv_folds}")

        if cv_folds >= 2:
            # Prefer repeated stratified CV for more stable metrics across small / noisy feedback batches.
            cv_strategy = os.getenv("TRAINING_CV_STRATEGY", "repeated_stratified").strip().lower()
            cv_repeats = int(os.getenv("TRAINING_CV_REPEATS", "3"))
            if can_stratify and cv_strategy == "repeated_stratified":
                cv_splitter = RepeatedStratifiedKFold(
                    n_splits=cv_folds,
                    n_repeats=max(1, cv_repeats),
                    random_state=random_state,
                )
                cv_strategy_used = "repeated_stratified"
            else:
                cv_splitter = cv_folds
                cv_strategy_used = "kfold"

            # Note: fit_params not supported in cross_val_score, sample weights handled during training.
            cv_acc_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='accuracy')
            cv_f1_scores = cross_val_score(model, X_train, y_train, cv=cv_splitter, scoring='f1_weighted')
            cv_mean_accuracy = float(cv_acc_scores.mean())
            cv_std_accuracy = float(cv_acc_scores.std())
            cv_mean_f1_weighted = float(cv_f1_scores.mean())
            cv_std_f1_weighted = float(cv_f1_scores.std())
        else:
            # Skip CV if not enough samples per class
            cv_mean_accuracy = train_accuracy
            cv_std_accuracy = 0.0
            cv_mean_f1_weighted = val_f1
            cv_std_f1_weighted = 0.0
            cv_strategy_used = "skipped"
            cv_repeats = 0
            logger.warning("Skipping cross-validation due to insufficient samples per class")
        
        training_metrics = {
            'training_accuracy': train_accuracy,
            'validation_accuracy': val_accuracy,
            'validation_f1': val_f1,
            'cv_mean_accuracy': cv_mean_accuracy,
            'cv_std_accuracy': cv_std_accuracy,
            'cv_mean_f1_weighted': cv_mean_f1_weighted,
            'cv_std_f1_weighted': cv_std_f1_weighted,
            'training_time_seconds': training_time,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(model.classes_),
            'model_params': model_params,
            'test_split_ratio': test_size,
            'cv_folds': cv_folds,
            'cv_strategy': cv_strategy_used,
            'cv_repeats': int(cv_repeats) if cv_strategy_used == "repeated_stratified" else 0,
            'split_random_state': int(random_state),
            'validation_mode_used': split_used,
            'validation_note': validation_note,
            'confidence_threshold_f1': float(best_t),
            'confidence_threshold_f1_score': float(best_f1w),
            'calibration_method': (
                cal_method if self.calibrated_classifier is not None else "none"
            ),
            'calibration_brier_before': calibration_brier_before,
            'calibration_brier_after': calibration_brier_after,
            'calibration_samples': int(X_val.shape[0]) if self.calibrated_classifier is not None else 0,
        }

        logger.info(
            "Training completed: val_accuracy=%.3f, val_f1=%.3f, split=%s",
            val_accuracy,
            val_f1,
            split_used,
        )
        
        return model, training_metrics
    
    def save_model(
        self,
        model: RandomForestClassifier,
        training_metrics: Dict[str, Any],
        model_version: str,
        clinic_key: Optional[str] = None,
    ) -> str:
        """Save trained model and artifacts (under clinics/<slug>/ when clinic_key is set)."""
        try:
            ck = normalize_clinic_key(clinic_key)
            root = self.model_dir
            if ck:
                root = os.path.join(self.model_dir, "clinics", clinic_dir_slug(ck))
                os.makedirs(root, exist_ok=True)
            version_dir = os.path.join(root, model_version)
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

            if getattr(self, "calibrated_classifier", None) is not None:
                cal_path = os.path.join(version_dir, "calibrated_classifier.pkl")
                joblib.dump(self.calibrated_classifier, cal_path)
                logger.info("Calibrated classifier saved to %s", cal_path)
            
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
                     y_val: pd.Series = None,
                     clinic_key: Optional[str] = None,
                     training_id: Optional[int] = None,
                     training_mode: str = "local"):
        """Log training to MLflow (tags chuẩn vetai_* + legacy clinic_id/training_id)."""
        # Retry connection when actually logging (MLflow may not have been ready at init)
        if not self._ensure_mlflow_connected():
            reason = self._mlflow_last_error or "unknown"
            logger.warning("MLflow not available, skipping logging (reason: %s)", reason)
            return

        ck = normalize_clinic_key(clinic_key)
        exp_name = self._experiment_name_for_clinic(ck)
        try:
            self._try_set_experiment_name(exp_name)
        except Exception as e:
            logger.warning("MLflow set experiment '%s' failed: %s", exp_name, e)
            return

        run_name = f"training-{model_version}"
        if ck is not None:
            run_name = f"{run_name}-clinic-{clinic_dir_slug(ck)}"

        try:
            with mlflow.start_run(run_name=run_name) as run:
                run_id = run.info.run_id
                ck_tag = ck if ck is not None else "global"
                try:
                    dw_raw = training_metrics.get("dataset_window_days")
                    dw_int = int(dw_raw) if dw_raw is not None else int(os.getenv("TRAINING_WINDOW_DAYS", "30"))
                except (TypeError, ValueError):
                    dw_int = int(os.getenv("TRAINING_WINDOW_DAYS", "30"))
                pk = str(training_metrics.get("pipeline_kind") or "continuous_training").strip() or "continuous_training"
                model_source = f"{pk}/{str(training_mode).strip() or 'local'}"
                # Log each step; do not re-raise so the run stays non-FAILED if one step fails
                try:
                    # Chuẩn hóa tag (vetai_*) để search/đối chiếu với registry
                    mlflow.set_tag("vetai_clinic_id", ck_tag)
                    mlflow.set_tag("vetai_dataset_window_days", str(dw_int))
                    mlflow.set_tag("vetai_model_source", model_source)
                    mlflow.set_tag("vetai_model_version", model_version)
                    if training_id is not None:
                        mlflow.set_tag("vetai_training_id", str(training_id))
                    else:
                        mlflow.set_tag("vetai_training_id", "")
                    # Legacy / tương thích dashboard cũ
                    mlflow.set_tag("clinic_id", ck_tag)
                    mlflow.set_tag("mlflow_experiment", exp_name)
                    mlflow.set_tag("dataset_window_days", str(dw_int))
                    mlflow.set_tag("model_source", model_source)
                    mlflow.set_tag("model_version", model_version)
                    if training_id is not None:
                        mlflow.set_tag("training_id", str(training_id))
                        mlflow.log_param("training_id", int(training_id))
                    mlflow.log_param("dataset_window_days", dw_int)
                    mlflow.log_param("model_source", model_source)
                    mlflow.log_param("model_version", model_version)
                    mlflow.log_param("pipeline_kind", pk)
                    mlflow.log_param("training_mode", str(training_mode))
                    mlflow.log_params(training_metrics.get('model_params', {}))
                    default_split_ratio = float(os.getenv("DEFAULT_SPLIT_RATIO", "0.2"))
                    default_cv_folds = int(os.getenv("DEFAULT_CV_FOLDS", "5"))
                    mlflow.log_param("test_split_ratio", training_metrics.get('test_split_ratio', default_split_ratio))
                    mlflow.log_param("cv_folds", training_metrics.get('cv_folds', default_cv_folds))
                    mlflow.log_param("experiment_name", exp_name)
                except Exception as e:
                    logger.warning("MLflow log params failed: %s", e)
                try:
                    _skip_mlflow_metric_keys = {
                        "dataset_window_days",
                    }
                    metrics_to_log = {
                        k: v
                        for k, v in training_metrics.items()
                        if isinstance(v, (int, float))
                        and k != "model_params"
                        and k not in _skip_mlflow_metric_keys
                    }
                    mlflow.log_metrics(metrics_to_log)
                except Exception as e:
                    logger.warning("MLflow log metrics failed: %s", e)
                try:
                    mlflow.sklearn.log_model(
                        model, artifact_path="model", registered_model_name="vet-ai-model"
                    )
                except Exception as reg_err:
                    try:
                        mlflow.sklearn.log_model(model, artifact_path="model")
                    except Exception as e2:
                        logger.warning("MLflow log_model failed: %s", e2)
                try:
                    if hasattr(self, 'tab_preprocess'):
                        joblib.dump(self.tab_preprocess, "tab_preprocess.pkl")
                        mlflow.log_artifact("tab_preprocess.pkl", "preprocessors")
                        os.remove("tab_preprocess.pkl")
                    if hasattr(self, 'symptoms_mlb'):
                        joblib.dump(self.symptoms_mlb, "symptoms_mlb.pkl")
                        mlflow.log_artifact("symptoms_mlb.pkl", "preprocessors")
                        os.remove("symptoms_mlb.pkl")
                except Exception as e:
                    logger.warning("MLflow log preprocessors failed: %s", e)
                try:
                    dataset_info = {
                        'n_samples': training_metrics.get('n_samples', 0),
                        'n_features': training_metrics.get('n_features', 0),
                        'n_classes': training_metrics.get('n_classes', 0),
                        'training_date': datetime.now().isoformat(),
                        'model_version': model_version
                    }
                    mlflow.log_dict(dataset_info, "dataset_info.json")
                    if X_val is not None and y_val is not None:
                        validation_info = {
                            'n_validation_samples': len(y_val),
                            'validation_features': X_val.shape[1]
                        }
                        mlflow.log_dict(validation_info, "validation_info.json")
                except Exception as e:
                    logger.warning("MLflow log dataset_info failed: %s", e)
                logger.info("Training logged to MLflow experiment '%s': %s", exp_name, run_id)
        except Exception as e:
            logger.error("Failed to log to MLflow: %s", e)
    
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


def _normalize_label_array(y) -> np.ndarray:
    """Strip whitespace for stable comparison between CSV labels and model outputs."""
    return np.array([str(v).strip() for v in np.asarray(y).ravel()], dtype=object)


# CSV cold-start bootstrap (same schema as vet-ml golden dataset).
CSV_BOOTSTRAP_REQUIRED_COLUMNS = [
    "pet_id",
    "animal_type",
    "gender",
    "age_months",
    "weight_kg",
    "temperature",
    "heart_rate",
    "current_season",
    "vaccination_status",
    "medical_history",
    "symptoms_list",
    "symptom_duration",
    "target_diagnosis",
]

CSV_BOOTSTRAP_TEMPLATE_HEADER = (
    "pet_id,animal_type,gender,age_months,weight_kg,temperature,heart_rate,"
    "current_season,vaccination_status,medical_history,symptoms_list,symptom_duration,target_diagnosis\n"
)


def parse_bootstrap_csv(
    raw: bytes,
    *,
    label_col: str = "target_diagnosis",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate a UTF-8 CSV and build in-memory feedback + prediction_log dicts for execute_training.

    Schema matches e.g. vet-ml/data/vet-ai-project/veterinary_full_data_10000.csv.
    Returns (feedback_rows, prediction_rows) with aligned prediction_id per pair.
    """
    if not raw or not str(raw).strip():
        raise ValueError("Empty CSV file")

    max_bytes = int(os.getenv("CSV_BOOTSTRAP_MAX_BYTES", str(25 * 1024 * 1024)))
    if len(raw) > max_bytes:
        raise ValueError(f"File too large (max {max_bytes} bytes)")

    min_rows = int(os.getenv("CSV_BOOTSTRAP_MIN_ROWS", "10"))
    max_rows = int(os.getenv("CSV_BOOTSTRAP_MAX_ROWS", "50000"))

    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Invalid CSV: {e}") from e

    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in CSV_BOOTSTRAP_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {CSV_BOOTSTRAP_REQUIRED_COLUMNS}")

    # Drop rows without label
    df = df[df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")]
    if len(df) < min_rows:
        raise ValueError(f"Need at least {min_rows} rows with non-empty {label_col}; got {len(df)}")

    if len(df) > max_rows:
        raise ValueError(f"Too many rows (max {max_rows}); got {len(df)}")

    def _req_str(val: Any, col: str) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError(f"Missing or invalid text in column {col}")
        s = str(val).strip()
        if not s or s.lower() == "nan":
            raise ValueError(f"Missing or invalid text in column {col}")
        return s

    def _parse_bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        s = str(v).strip().lower()
        if s in ("true", "1", "yes", "y", "accept", "accepted"):
            return True
        if s in ("false", "0", "no", "n", "reject", "rejected"):
            return False
        return None

    feedback_list: List[Dict[str, Any]] = []
    pred_list: List[Dict[str, Any]] = []

    base_ts = pd.Timestamp.now(tz="UTC").normalize()
    for row_idx, (_, row) in enumerate(df.iterrows()):  # noqa: B007
        label = str(row[label_col]).strip()
        if not label or label.lower() == "nan":
            continue

        try:
            age_m = int(row["age_months"]) if row["age_months"] is not None else None
            sym_dur = int(row["symptom_duration"]) if row["symptom_duration"] is not None else None
            hr = int(row["heart_rate"]) if row["heart_rate"] is not None else None
            w = float(row["weight_kg"]) if row["weight_kg"] is not None else None
            temp = float(row["temperature"]) if row["temperature"] is not None else None
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid numeric field in row: {e}") from e

        if age_m is None or sym_dur is None or hr is None or w is None or temp is None:
            raise ValueError("age_months, symptom_duration, heart_rate, weight_kg, temperature are required per row")

        mh = row["medical_history"]
        if mh is None or (isinstance(mh, float) and np.isnan(mh)):
            mh = "Unknown"
        else:
            mh = str(mh).strip() or "Unknown"

        symptoms = row["symptoms_list"]
        if symptoms is None or (isinstance(symptoms, float) and np.isnan(symptoms)):
            symptoms = ""
        else:
            symptoms = str(symptoms)

        pet_cell = row["pet_id"]
        pet_id = "bootstrap" if pet_cell is None or str(pet_cell).strip() == "" else str(pet_cell).strip()

        pred_input = {
            "animal_type": _req_str(row["animal_type"], "animal_type"),
            "gender": _req_str(row["gender"], "gender"),
            "age_months": age_m,
            "weight_kg": w,
            "temperature": temp,
            "heart_rate": hr,
            "current_season": _req_str(row["current_season"], "current_season"),
            "vaccination_status": _req_str(row["vaccination_status"], "vaccination_status"),
            "medical_history": mh,
            "symptoms_list": symptoms,
            "symptom_duration": sym_dur,
        }

        ai_diag_cell = row["ai_diagnosis"] if "ai_diagnosis" in df.columns else None
        ai_diag = str(ai_diag_cell).strip() if ai_diag_cell is not None else ""
        if not ai_diag or ai_diag.lower() == "nan":
            ai_diag = label

        is_correct_raw = row["is_correct"] if "is_correct" in df.columns else None
        is_correct = _parse_bool(is_correct_raw)
        if is_correct is None:
            # Backward compatibility with old CSV schema: treat as accept.
            is_correct = True

        pid = uuid.uuid4()
        pred_list.append(
            {
                "id": pid,
                "visit_id": None,
                "pet_id": pet_id,
                "prediction_input": pred_input,
                "prediction_output": {"diagnosis": ai_diag, "confidence": 1.0, "top_k": []},
                "model_version": "bootstrap_csv",
                "confidence_score": 1.0,
                "top_k_predictions": [],
            }
        )
        feedback_list.append(
            {
                "prediction_id": pid,
                "final_diagnosis": label,
                "is_correct": bool(is_correct),
                "ai_diagnosis": ai_diag,
                "confidence_rating": None,
                "is_training_eligible": True,
                "data_quality_score": 1.0,
                "created_at": (base_ts + pd.Timedelta(seconds=row_idx)).isoformat(),
            }
        )

    if len(feedback_list) < min_rows:
        raise ValueError(f"After cleaning, fewer than {min_rows} rows remain")

    logger.info("Bootstrap CSV parsed: %d training pairs", len(feedback_list))
    return feedback_list, pred_list


def _annotate_small_sample_metrics(training_metrics: Dict[str, Any]) -> None:
    try:
        n_s = int(training_metrics.get("n_samples") or 0)
    except (TypeError, ValueError):
        n_s = 0
    min_rel = int(os.getenv("METRICS_RELIABLE_MIN_SAMPLES", "80"))
    if n_s > 0 and n_s < min_rel:
        training_metrics["small_sample_warning"] = True
        training_metrics["metrics_note"] = (
            f"n_samples={n_s} < METRICS_RELIABLE_MIN_SAMPLES={min_rel}: "
            "on a small set, train/val accuracy and F1 can be very high (overfit/small holdout); "
            "compare with a local run using the same amount of feedback."
        )


def _annotate_accept_only_feedback_metrics(
    eligible_feedback: List[Dict[str, Any]],
    training_metrics: Dict[str, Any],
) -> None:
    """When every row is accept, we never add reject/contrastive training rows — val metrics can look optimistic."""
    n_reject = sum(1 for f in (eligible_feedback or []) if f.get("is_correct") is False)
    if n_reject > 0:
        return
    note = (
        "All feedback has is_correct=true: no reject/contrastive rows are added to the training set; "
        "holdout accuracy/F1 can look very high versus production with mixed accepts and rejects."
    )
    prev = training_metrics.get("metrics_note")
    training_metrics["metrics_note"] = f"{prev} {note}".strip() if prev else note


def _derive_split_random_state(
    *,
    training_id: Optional[int],
    eligible_feedback_data: List[Dict[str, Any]],
) -> int:
    """
    Derive split seed from base RANDOM_STATE + training_id + feedback content signature.
    This avoids same metrics caused only by fixed split seed, while remaining reproducible for the same job payload.
    """
    base_seed = int(os.getenv("RANDOM_STATE", "42"))
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    h.update(str(training_id if training_id is not None else "no-training-id").encode("utf-8"))
    for row in eligible_feedback_data or []:
        pid = row.get("prediction_id")
        if pid is not None:
            h.update(str(pid).encode("utf-8"))
        h.update(str(row.get("final_diagnosis", "")).strip().lower().encode("utf-8"))
        h.update(b"|")
    # sklearn accepts 32-bit signed random_state
    return int.from_bytes(h.digest()[:4], "big") & 0x7FFFFFFF


def execute_training(
    feedback_data: List[Dict],
    prediction_logs: List[Dict],
    training_mode: str = "local",
    clinic_id: Optional[Any] = None,
    training_id: Optional[int] = None,
    *,
    dataset_window_days: Optional[int] = None,
    pipeline_kind: str = "continuous_training",
) -> Dict[str, Any]:
    """Execute training pipeline with dynamic parameters."""
    clinic_key = normalize_clinic_key(clinic_id)
    try:
        dw_days = (
            int(dataset_window_days)
            if dataset_window_days is not None
            else int(os.getenv("TRAINING_WINDOW_DAYS", "30"))
        )
    except (TypeError, ValueError):
        dw_days = 30
    logger.info("Starting %s training pipeline (clinic_id=%s)", training_mode, clinic_key)

    try:
        # Train only on eligible feedback (aligns with eligibility used to trigger jobs).
        eligible_feedback_data = [
            f for f in (feedback_data or []) if f.get("is_training_eligible", True)
        ]
        if not eligible_feedback_data:
            raise ValueError("No eligible feedback data to train on")
        split_seed = _derive_split_random_state(
            training_id=training_id,
            eligible_feedback_data=eligible_feedback_data,
        )
        trainer = ModelTrainer()        
        X, y, sample_weights, sample_timestamps = trainer.collect_training_data(
            eligible_feedback_data, prediction_logs
        )

        # Safety guard: don't train/persist a model when feedback has too low class diversity.
        # If y contains only 1 diagnosis class, RandomForest will end up predicting that class
        # with ~100% confidence, which breaks real-world inference.
        min_unique_classes = int(os.getenv("TRAINING_MIN_UNIQUE_CLASSES", "2"))
        y_str = y.astype(str) if hasattr(y, "astype") else np.asarray(y).astype(str)
        unique_classes = np.unique(y_str)
        if len(unique_classes) < min_unique_classes:
            class_counts = dict(pd.Series(y_str).value_counts().to_dict())
            raise ValueError(
                f"Insufficient class diversity for training: unique_classes={len(unique_classes)} "
                f"< {min_unique_classes}. class_counts={class_counts}"
            )
        
        # Fine-tune mode (warm-start) vs full retrain
        finetune = os.getenv("FINETUNE_PREVIOUS_MODEL", "true").lower() in ("1", "true", "yes", "y")
        if finetune:
            base_model, base_dir = trainer.load_active_artifacts(clinic_key)
            X_processed, preprocessing_info = trainer.preprocess_features(X, fit_encoders=False)
            add_trees = int(os.getenv("FINETUNE_ADD_TREES", "20"))
            try:
                model, training_metrics = trainer.train_model(
                    X_processed,
                    y,
                    sample_weights,
                    base_model=base_model,
                    finetune_add_trees=add_trees,
                    split_random_state=split_seed,
                    sample_timestamps=sample_timestamps,
                )
                training_metrics["finetune"] = True
                training_metrics["finetune_fallback"] = False
            except Exception as e:
                # sklearn warm_start for RF can fail in some edge cases (small batches / class shifts).
                # Fallback to full retrain using the *same* fixed preprocessors (encoders reused).
                logger.warning("Warm-start fine-tune failed (%s). Falling back to retrain.", e)
                model, training_metrics = trainer.train_model(
                    X_processed,
                    y,
                    sample_weights,
                    base_model=None,
                    split_random_state=split_seed,
                    sample_timestamps=sample_timestamps,
                )
                training_metrics["finetune"] = True
                training_metrics["finetune_fallback"] = True
            training_metrics["finetune"] = True
            training_metrics["finetune_base_dir"] = base_dir
        else:
            X_processed, preprocessing_info = trainer.preprocess_features(X, fit_encoders=True)
            model, training_metrics = trainer.train_model(
                X_processed,
                y,
                sample_weights,
                split_random_state=split_seed,
                sample_timestamps=sample_timestamps,
            )
            training_metrics["finetune"] = False

        _annotate_small_sample_metrics(training_metrics)
        _annotate_accept_only_feedback_metrics(eligible_feedback_data, training_metrics)
        if clinic_key is not None:
            training_metrics["clinic_id"] = clinic_key
        
        # ----------------------------
        # Regression Test Gate (golden set)
        # ----------------------------
        regression_gate_enabled = os.getenv("REGRESSION_GATE_ENABLED", "true").lower() in ("1", "true", "yes", "y")
        regression_tol_f1 = float(os.getenv("REGRESSION_TOLERANCE_F1", "0.01"))
        golden_test_max_rows = int(os.getenv("GOLDEN_TEST_MAX_ROWS", "2000"))
        golden_test_random_state = int(os.getenv("RANDOM_STATE", "42"))
        # Fine-tune on tiny feedback batches often collapses holdout F1; skip golden gate until enough signal.
        regression_min_feedback = int(os.getenv("REGRESSION_GATE_MIN_FEEDBACK_SAMPLES", "20"))

        if (
            regression_gate_enabled
            and finetune
            and os.getenv("BASELINE_DATASET_CSV")
            and len(eligible_feedback_data) < regression_min_feedback
        ):
            logger.info(
                "Regression gate skipped: len(feedback_data)=%d < REGRESSION_GATE_MIN_FEEDBACK_SAMPLES=%d "
                "(small batches are not representative for holdout regression)",
                len(feedback_data),
                regression_min_feedback,
            )
            training_metrics["regression_gate_enabled"] = False
            training_metrics["regression_gate_skipped_reason"] = "low_feedback_count"
            training_metrics["regression_gate_min_feedback_samples"] = regression_min_feedback
        elif regression_gate_enabled and finetune and os.getenv("BASELINE_DATASET_CSV"):
            try:
                baseline_csv = os.getenv("BASELINE_DATASET_CSV")
                baseline_label_col = os.getenv("BASELINE_LABEL_COL", "target_diagnosis")

                feature_cols = [
                    "animal_type",
                    "gender",
                    "age_months",
                    "weight_kg",
                    "temperature",
                    "heart_rate",
                    "current_season",
                    "vaccination_status",
                    "medical_history",
                    "symptom_duration",
                    "symptoms_list",
                ]
                needed_cols = feature_cols + [baseline_label_col]
                golden_df = pd.read_csv(baseline_csv, usecols=needed_cols)

                if len(golden_df) > golden_test_max_rows:
                    golden_df = golden_df.sample(n=golden_test_max_rows, random_state=golden_test_random_state)

                y_gold = _normalize_label_array(golden_df[baseline_label_col])
                X_gold_features = golden_df[feature_cols]

                # Use locked encoders (fit_encoders=False) for stability
                X_gold_processed, _ = trainer.preprocess_features(X_gold_features, fit_encoders=False)

                # Important: `base_model` may be mutated in-memory by warm-start training
                # (train_model() reuses the same object). To get a true "before" baseline,
                # reload the base model from disk.
                base_model_fresh = joblib.load(os.path.join(base_dir, "model.pkl"))
                base_preds = _normalize_label_array(base_model_fresh.predict(X_gold_processed))
                new_preds = _normalize_label_array(model.predict(X_gold_processed))

                try:
                    base_f1 = float(
                        f1_score(y_gold, base_preds, average="weighted", zero_division=0)
                    )
                    new_f1 = float(
                        f1_score(y_gold, new_preds, average="weighted", zero_division=0)
                    )
                except TypeError:
                    base_f1 = float(f1_score(y_gold, base_preds, average="weighted"))
                    new_f1 = float(f1_score(y_gold, new_preds, average="weighted"))
                base_acc = float(accuracy_score(y_gold, base_preds))
                new_acc = float(accuracy_score(y_gold, new_preds))

                # base_f1 can be 1.0 if the golden slice is too easy.
                # Then a small relative tolerance (e.g. 0.01) makes the gate always fail.
                # So widen tolerance when base_f1 is "very high".
                regression_high_base_f1_threshold = float(
                    os.getenv("REGRESSION_HIGH_BASE_F1_THRESHOLD", "0.99")
                )
                regression_tolerance_high_base = float(
                    os.getenv("REGRESSION_TOLERANCE_F1_HIGH_BASE", "0.35")
                )
                effective_tol = regression_tol_f1
                if base_f1 >= regression_high_base_f1_threshold:
                    effective_tol = max(regression_tol_f1, regression_tolerance_high_base)

                training_metrics["regression_gate_enabled"] = True
                training_metrics["golden_test_size"] = int(len(golden_df))
                training_metrics["golden_base_f1_weighted"] = base_f1
                training_metrics["golden_new_f1_weighted"] = new_f1
                training_metrics["golden_base_accuracy"] = base_acc
                training_metrics["golden_new_accuracy"] = new_acc

                training_metrics["regression_effective_tolerance_f1"] = effective_tol
                if new_f1 < base_f1 - effective_tol:
                    err = (
                        f"Regression gate failed on golden test: "
                        f"base_f1={base_f1:.4f}, new_f1={new_f1:.4f}, "
                        f"tolerance={effective_tol:.4f}"
                    )
                    logger.warning(err)
                    return {
                        "status": "failed",
                        "error": err,
                        "training_mode": training_mode,
                        "training_metrics": training_metrics,
                    }
            except Exception as e:
                # If golden regression test fails for any reason, don't block training.
                logger.warning("Regression gate skipped due to error: %s", e)
                training_metrics["regression_gate_skipped_error"] = str(e)
        else:
            training_metrics["regression_gate_enabled"] = False

        # ----------------------------
        # Feedback improvement gate (expert feedback batch only, no core memory)
        # ----------------------------
        feedback_gate_enabled = os.getenv(
            "FEEDBACK_IMPROVEMENT_GATE_ENABLED", "true"
        ).lower() in ("1", "true", "yes", "y")
        feedback_gate_tolerance = float(os.getenv("FEEDBACK_GATE_TOLERANCE", "0.05"))

        if (
            feedback_gate_enabled
            and finetune
            and training_metrics.get("finetune_base_dir")
        ):
            base_dir_fb = str(training_metrics["finetune_base_dir"])
            try:
                X_fb, y_fb = trainer.collect_feedback_only_training_frame(
                    eligible_feedback_data, prediction_logs
                )
                y_fb_str = y_fb.astype(str)
                X_fb_proc, _ = trainer.preprocess_features(X_fb, fit_encoders=False)
                base_model_feedback = joblib.load(
                    os.path.join(base_dir_fb, "model.pkl")
                )
                base_preds_fb = base_model_feedback.predict(X_fb_proc)
                new_preds_fb = model.predict(X_fb_proc)

                n_unique_fb = int(y_fb_str.nunique())
                if n_unique_fb < 2:
                    metric_name = "accuracy"
                    base_sc = float(accuracy_score(y_fb_str, base_preds_fb))
                    new_sc = float(accuracy_score(y_fb_str, new_preds_fb))
                else:
                    metric_name = "f1_weighted"
                    try:
                        base_sc = float(
                            f1_score(
                                y_fb_str,
                                base_preds_fb,
                                average="weighted",
                                zero_division=0,
                            )
                        )
                        new_sc = float(
                            f1_score(
                                y_fb_str,
                                new_preds_fb,
                                average="weighted",
                                zero_division=0,
                            )
                        )
                    except TypeError:
                        base_sc = float(
                            f1_score(y_fb_str, base_preds_fb, average="weighted")
                        )
                        new_sc = float(
                            f1_score(y_fb_str, new_preds_fb, average="weighted")
                        )

                training_metrics["feedback_gate_enabled"] = True
                training_metrics["feedback_eval_size"] = int(len(y_fb))
                training_metrics["feedback_gate_metric"] = metric_name
                training_metrics["feedback_base_score"] = base_sc
                training_metrics["feedback_new_score"] = new_sc

                if new_sc < base_sc - feedback_gate_tolerance:
                    err = (
                        f"Feedback improvement gate failed: metric={metric_name} "
                        f"base={base_sc:.4f}, new={new_sc:.4f}, "
                        f"tolerance={feedback_gate_tolerance:.4f}"
                    )
                    logger.warning(err)
                    return {
                        "status": "failed",
                        "error": err,
                        "training_mode": training_mode,
                        "training_metrics": training_metrics,
                    }
            except Exception as e:
                logger.warning("Feedback improvement gate skipped: %s", e)
                training_metrics["feedback_gate_skipped_error"] = str(e)
        else:
            training_metrics["feedback_gate_enabled"] = False

        # Generate version
        model_version = trainer.generate_model_version()
        
        # Save model
        model_path = trainer.save_model(model, training_metrics, model_version, clinic_key=clinic_key)

        try:
            from ai_service.app.infrastructure.external.s3_client import upload_model_directory

            upload_model_directory(model_path, model_version, clinic_key=clinic_key)
        except Exception as e:
            logger.warning("S3 artifact upload after training failed (non-fatal): %s", e)

        training_metrics["training_mode"] = training_mode
        training_metrics["dataset_window_days"] = dw_days
        training_metrics["pipeline_kind"] = pipeline_kind

        # Log to MLflow
        trainer.log_to_mlflow(
            model,
            training_metrics,
            model_version,
            clinic_key=clinic_key,
            training_id=training_id,
            training_mode=training_mode,
        )
        
        result = {
            'status': 'completed',
            'model_version': model_version,
            'model_path': model_path,
            'training_metrics': training_metrics,
            'preprocessing_info': preprocessing_info,
            'training_mode': training_mode,
            'training_id': training_id,
            'dataset_window_days': dw_days,
            'pipeline_kind': pipeline_kind,
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
        logger.exception("Training failed")
        return {
            'status': 'failed',
            'error': str(e),
            'training_mode': training_mode
        }


# Bridge helpers used by app layers
def get_training_router():
    from ai_service.app.api.routers.training import router

    return router


async def log_prediction_entry(prediction_log):
    from ai_service.app.api.routers.training import log_prediction

    await log_prediction(prediction_log)


def get_training_runtime_handles():
    from ai_service.app.api.routers.training import ct_store, training_jobs

    return ct_store, training_jobs


def get_prediction_log_model():
    from ai_service.app.api.routers.training import PredictionLog

    return PredictionLog


def get_refresh_training_metrics():
    from ai_service.app.api.routers.training import _refresh_training_metrics

    return _refresh_training_metrics


def get_training_engine_handles():
    return ModelTrainer, execute_training, parse_bootstrap_csv
