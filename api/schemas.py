from pydantic import BaseModel, Field
from typing import List

class RecommendRequest(BaseModel):
    product_name: str = Field(..., min_length=1)
    catalog: List[str] = Field(..., min_items=1)
    top_n: int = Field(default=5, ge=1, le=20)

class RecommendResponse(BaseModel):
    product_name: str
    recommendations: List[dict]
    total_catalog_size: int
