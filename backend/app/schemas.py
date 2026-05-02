import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# Gemini sometimes sends qualitative storey bands instead of canonical HDB strings.
_QUALITATIVE_STOREY = {
    "LOW": "04 TO 06",
    "LOW FLOOR": "04 TO 06",
    "GROUND": "01 TO 03",
    "GROUND FLOOR": "01 TO 03",
    "MID": "13 TO 15",
    "MIDDLE": "13 TO 15",
    "MID FLOOR": "13 TO 15",
    "HIGH": "22 TO 24",
    "HIGH FLOOR": "22 TO 24",
}


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

    model_config = {"extra": "ignore"}

    @field_validator("floor_area_sqm", mode="before")
    @classmethod
    def coerce_floor_area_sqm(cls, v: Any) -> Any:
        """Accept ints/floats and messy strings ('93 sqm', '~90', '93.5 sq metres')."""
        if isinstance(v, bool):
            raise ValueError("floor_area_sqm must be numeric")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip().lower().replace(",", "")
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if m:
                return float(m.group(1))
        raise ValueError("floor_area_sqm must be a number")

    @field_validator("remaining_lease_years", mode="before")
    @classmethod
    def coerce_remaining_lease(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, str) and not v.strip():
            return None
        if isinstance(v, str):
            try:
                return float(v.strip().replace(",", ""))
            except ValueError:
                return None
        if isinstance(v, (int, float)):
            return float(v)
        return None

    @field_validator("storey_range")
    @classmethod
    def validate_storey(cls, v: str) -> str:
        from app.constants import VALID_STOREY_RANGES
        v = v.strip().upper()
        v = re.sub(r"\s*-\s*", " TO ", v)
        v = re.sub(r"\s+", " ", v).strip()
        if v in _QUALITATIVE_STOREY:
            v = _QUALITATIVE_STOREY[v]
        if v in VALID_STOREY_RANGES:
            return v
        # LLMs often emit "7 TO 9" instead of zero-padded canonical bands
        m = re.match(r"^(\d{1,2})\s+TO\s+(\d{1,2})$", v)
        if m:
            low, high = int(m.group(1)), int(m.group(2))
            padded = f"{low:02d} TO {high:02d}"
            if padded in VALID_STOREY_RANGES:
                return padded
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
