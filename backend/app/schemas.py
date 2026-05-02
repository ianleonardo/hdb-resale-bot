import re
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Minimal bot payload: town + block + storey + floor area.
    Flat type/model and lease default server-side when omitted."""

    town:           str
    block:          str
    storey_range:   str    # "07 TO 09"
    floor_area_sqm: float = Field(ge=20.0, le=300.0)

    flat_type:             Optional[str]   = None
    flat_model:            Optional[str]   = None
    remaining_lease_years: Optional[float] = Field(default=None, ge=0.0, le=99.0)
    street_name:           Optional[str]   = None

    @field_validator("storey_range")
    @classmethod
    def validate_storey(cls, v: str) -> str:
        from app.constants import VALID_STOREY_RANGES
        v = v.strip().upper()
        if v in VALID_STOREY_RANGES:
            return v
        # Snap a bare floor number to the nearest valid band
        if re.match(r"^\d+$", v):
            floor = int(v)
            low = ((floor - 1) // 3) * 3 + 1
            snapped = f"{low:02d} TO {low + 2:02d}"
            if snapped in VALID_STOREY_RANGES:
                return snapped
        raise ValueError(f"storey_range must be one of {VALID_STOREY_RANGES}")

    @field_validator("town", "block")
    @classmethod
    def uppercase_required(cls, v: str) -> str:
        v = v.strip().upper()
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("flat_type", "flat_model")
    @classmethod
    def strip_optional_flat(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

    @field_validator("street_name")
    @classmethod
    def uppercase_optional_street(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().upper()
        return v if v else None


class PredictResponse(BaseModel):
    predicted_price: float
    price_range:     dict
    confidence:      str
    model_version:   str
    input_echo:      dict
