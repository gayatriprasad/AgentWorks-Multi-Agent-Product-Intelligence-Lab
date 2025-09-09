# app/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

# -------------------------
# Search Products
# -------------------------
class SearchProductsIn(BaseModel):
    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of products to return")

class Product(BaseModel):
    id: str
    name: str
    brand: Optional[str] = None
    price: Optional[float] = None
    url: Optional[str] = None

class SearchProductsOut(BaseModel):
    items: List[Product]

# -------------------------
# Compare Prices
# -------------------------
class ComparePricesIn(BaseModel):
    product_id: str

class PriceQuote(BaseModel):
    vendor: str
    price: float
    currency: str = "INR"
    url: Optional[str] = None

class ComparePricesOut(BaseModel):
    quotes: List[PriceQuote]

# -------------------------
# Fetch Reviews
# -------------------------
class FetchReviewsIn(BaseModel):
    product_id: str
    top_k: int = 5

class Review(BaseModel):
    author: Optional[str] = None
    rating: Optional[float] = None
    text: str

class FetchReviewsOut(BaseModel):
    reviews: List[Review]

# -------------------------
# Answer FAQ
# -------------------------
class AnswerFAQIn(BaseModel):
    question: str

class AnswerFAQOut(BaseModel):
    answer: str
