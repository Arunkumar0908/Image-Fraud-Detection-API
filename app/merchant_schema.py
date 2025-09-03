from pydantic import BaseModel, Field
from typing import Dict, Any

class Merchant(BaseModel):
    merchant_id: str = Field(..., max_length=36)  
    name: str = Field(..., max_length=50)
    contact: str = Field(..., max_length=20)
    latitude: float
    longitude: float
    location: Dict[str, Any]
    front: Dict[str, Any]
    nameboard: Dict[str, Any]
    inside: Dict[str, Any]
