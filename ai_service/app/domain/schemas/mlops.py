from pydantic import BaseModel, Field


class ActiveModelRequest(BaseModel):
    model_version: str = Field(..., description="Model version to activate for inference")
