from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def parse_symptoms(symptoms: Any) -> list[str]:
    if symptoms is None:
        return []
    text = str(symptoms).strip()
    if not text:
        return []
    return [item for item in (x.strip().lower() for x in text.split(",")) if item]


class PredictRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    animal_type: str = Field(..., description="Dog or cat only (case-insensitive): dog or cat.")
    gender: str
    age_months: int
    weight_kg: float
    temperature: float
    heart_rate: int
    current_season: str
    vaccination_status: str
    medical_history: str | None = "Unknown"
    symptoms_list: str
    symptom_duration: int
    clinic_id: int | str | None = Field(
        default=None,
        alias="clinicId",
        description="Clinic id (legacy int for pinned model dirs) or UUID string.",
    )
    pet_id: str | None = Field(default=None, alias="petId")
    visit_id: int | str | None = Field(default=None, alias="visitId")
    model_version: str | None = Field(
        default=None,
        alias="modelVersion",
        description=(
            "Optional model version folder (e.g. v2.0). When omitted, the clinic active model "
            "(or global default) is used. Must be listed by GET /predict/models for the same clinic."
        ),
    )

    @field_validator("animal_type")
    @classmethod
    def _animal_dog_or_cat_only(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in ("dog", "cat"):
            raise ValueError("animal_type must be 'dog' or 'cat'")
        return normalized
