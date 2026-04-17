import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    # Required
    town:              str
    flat_type:         str
    flat_model:        str
    storey_range:      str           # "07 TO 09"
    floor_area_sqm:    float = Field(ge=20.0, le=300.0)

    # Optional — model uses defaults when absent
    remaining_lease_years: Optional[float] = Field(default=None, ge=0.0, le=99.0)
    street_name:           Optional[str]   = None
    block:                 Optional[str]   = None

    @field_validator("storey_range")
    @classmethod
    def validate_storey(cls, v: str) -> str:
        if not re.match(r"^\d{2} TO \d{2}$", v.strip()):
            raise ValueError("storey_range must match 'NN TO NN'")
        return v.strip().upper()

    @field_validator("town", "flat_type", "flat_model")
    @classmethod
    def uppercase_str(cls, v: str) -> str:
        return v.strip().upper()


class PredictResponse(BaseModel):
    predicted_price: float
    price_range:     dict
    confidence:      str
    model_version:   str
    input_echo:      dict
