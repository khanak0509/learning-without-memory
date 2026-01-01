from pydantic import BaseModel, Field, field_validator

class Schema(BaseModel):
    score: float = Field(..., description="Clarity score between 0 and 1")
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {v}")
        return v 