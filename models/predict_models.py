from pydantic import BaseModel
from typing import Optional, List, Dict

class PredictRequest(BaseModel):
    product_type: str
    weight_kg: float
    energy_mix_pct_renewables: Optional[float] = None
    co2_kg_per_kg: Optional[float] = None
    recycled_content_pct: Optional[float] = None
    lifetime_years: Optional[float] = None
    reuse_probability_pct: Optional[float] = None

class PredictResponse(BaseModel):
    success: bool
    data: Dict[str, float]
    predicted_fields: List[str]
    confidence_scores: Dict[str, float] = {}
