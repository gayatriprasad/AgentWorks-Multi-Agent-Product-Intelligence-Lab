from pydantic import BaseModel, Field
from typing import List, Optional

class Product(BaseModel):
    product_id: str
    title: str
    category: Optional[str] = None
    price: Optional[float] = None
    description: Optional[str] = None

class Review(BaseModel):
    review_id: str
    product_id: str
    rating: Optional[float] = None
    text: str

class Recommendation(BaseModel):
    product: Product
    score: float
    summary: Optional[str] = None
    pros: Optional[List[str]] = None
    cons: Optional[List[str]] = None
