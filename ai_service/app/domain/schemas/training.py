from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictionLog(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: uuid.UUID
    visit_id: Optional[str | int] = None
    pet_id: str
    prediction_input: Dict
    prediction_output: Dict
    model_version: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    top_k_predictions: List[Dict]
    veterinarian_id: Optional[str] = None
    clinic_id: Optional[str] = None


class DoctorFeedback(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prediction_id: uuid.UUID
    final_diagnosis: str
    is_correct: bool = Field(
        ...,
        validation_alias=AliasChoices("is_correct", "isCorrect"),
        description="True nếu bác sĩ đồng ý với gợi ý AI; client có thể gửi isCorrect (camelCase).",
    )
    ai_diagnosis: Optional[str] = None
    confidence_rating: Optional[int] = Field(ge=0, le=5, default=None)
    comments: Optional[str] = None
    veterinarian_id: Optional[str] = None
    is_training_eligible: bool = True
    data_quality_score: float = Field(ge=0.0, le=1.0, default=1.0)
    training_pool: Optional[str] = Field(default=None, pattern="^(GLOBAL|CLINIC_ONLY)$")

    @field_validator("final_diagnosis", "ai_diagnosis", mode="before")
    @classmethod
    def _normalize_diag_text(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @model_validator(mode="after")
    def _validate_feedback_consistency(self) -> "DoctorFeedback":
        if not self.final_diagnosis:
            raise ValueError("final_diagnosis is required.")
        if self.ai_diagnosis:
            same = self.final_diagnosis.lower() == self.ai_diagnosis.lower()
            if self.is_correct and not same:
                raise ValueError(
                    "Invalid accept: final_diagnosis must equal ai_diagnosis. "
                    "Send is_correct=false with a different diagnosis if you disagree."
                )
            if (not self.is_correct) and same:
                raise ValueError(
                    "Invalid reject: final_diagnosis equals ai_diagnosis. "
                    "Choose a different diagnosis or set is_correct=true if you agree with the AI."
                )
        return self


class TrainingTriggerRequest(BaseModel):
    trigger_type: str = Field(pattern="^(scheduled|manual|automatic)$")
    trigger_reason: Optional[str] = None
    force: bool = False
    training_mode: str = Field(default="local", pattern="^(local|eks_hybrid)$")
    eks_node_group: Optional[str] = None
    clinic_id: Optional[str] = None

    @field_validator("clinic_id", mode="before")
    @classmethod
    def _normalize_trigger_clinic(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None


class TrainingStatus(BaseModel):
    model_config = ConfigDict(extra="ignore")

    training_id: int
    status: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_predictions: int
    eligible_feedback_count: int
    trigger_type: Optional[str] = None
    dataset_row_count: Optional[int] = None
    previous_model_version: Optional[str]
    new_model_version: Optional[str]
    training_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    f1_score: Optional[float]
    is_deployed: bool
    error_message: Optional[str]
    small_sample_warning: Optional[bool] = None
    metrics_note: Optional[str] = None
    promote_guardrail_passed: Optional[bool] = None
    promote_guardrail_reason: Optional[str] = None
    audit_snapshot: Optional[Dict[str, Any]] = None


class TrainingPolicyUpdate(BaseModel):
    training_threshold: int = Field(ge=1)
    training_window_days: int = Field(ge=1)
