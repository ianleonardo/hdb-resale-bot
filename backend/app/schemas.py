import re
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    town:                   str
    flat_type:              str
    flat_model:             str
    storey_range:           str           # "07 TO 09"
    floor_area_sqm:         float = Field(ge=20.0,  le=300.0)
    remaining_lease_years:  float = Field(ge=0.0,   le=99.0)
    street_name:            str
    block:                  str

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
