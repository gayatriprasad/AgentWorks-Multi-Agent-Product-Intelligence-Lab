# app/tools.py
from typing import Callable, Dict, Any
from pydantic import BaseModel
from app.schemas import (
    SearchProductsIn, SearchProductsOut,
    ComparePricesIn, ComparePricesOut,
    FetchReviewsIn, FetchReviewsOut,
    AnswerFAQIn, AnswerFAQOut,
)

# --------------------------------------------------------------------
# Tool spec / registry class
# --------------------------------------------------------------------
class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._specs: Dict[str, ToolSpec] = {}

    def register(self, name: str, func: Callable, spec: ToolSpec | None = None):
        self._tools[name] = func
        if spec:
            self._specs[name] = spec

    def list_specs(self):
        return list(self._specs.values())

    def call(self, name: str, **kwargs):
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name](**kwargs)

# --------------------------------------------------------------------
# Example tool implementations (replace stubs with your real code)
# --------------------------------------------------------------------
def search_products(query: str, top_k: int = 10) -> dict:
    # Replace with SearchIndex().search(query, top_k)
    items = [{
        "id": "p1",
        "name": "Demo Phone",
        "brand": "Acme",
        "price": 9999.0,
        "url": "https://example.com/p1",
    }]
    return SearchProductsOut(items=items).model_dump()

def compare_prices(product_id: str) -> dict:
    quotes = [
        {"vendor": "Amazon", "price": 9899.0, "currency": "INR", "url": "https://amazon.in/demo"},
        {"vendor": "Flipkart", "price": 9799.0, "currency": "INR", "url": "https://flipkart.com/demo"},
    ]
    return ComparePricesOut(quotes=quotes).model_dump()

def fetch_reviews(product_id: str, top_k: int = 5) -> dict:
    reviews = [
        {"author": "Rags", "rating": 4.5, "text": "Solid budget pick."},
        {"author": "Ani", "rating": 4.0, "text": "Camera could be better."},
    ][: top_k]
    return FetchReviewsOut(reviews=reviews).model_dump()

def answer_faq(question: str) -> dict:
    return AnswerFAQOut(answer=f"(Stub) Answer to: {question}").model_dump()

# --------------------------------------------------------------------
# Global registry + FUNCTION_REGISTRY alias (for agents/orchestrator)
# --------------------------------------------------------------------
registry = ToolRegistry()

registry.register("search_products", search_products, ToolSpec(
    name="search_products",
    description="Search catalog with natural language.",
    parameters=SearchProductsIn.model_json_schema(),
))

registry.register("compare_prices", compare_prices, ToolSpec(
    name="compare_prices",
    description="Get live price quotes for a product.",
    parameters=ComparePricesIn.model_json_schema(),
))

registry.register("fetch_reviews", fetch_reviews, ToolSpec(
    name="fetch_reviews",
    description="Fetch recent user reviews for a product.",
    parameters=FetchReviewsIn.model_json_schema(),
))

registry.register("answer_faq", answer_faq, ToolSpec(
    name="answer_faq",
    description="Answer a frequently asked question from the knowledge base.",
    parameters=AnswerFAQIn.model_json_schema(),
))

# For compatibility with agents.py
FUNCTION_REGISTRY: Dict[str, Callable] = {
    "search_products": search_products,
    "compare_prices": compare_prices,
    "fetch_reviews": fetch_reviews,
    "answer_faq": answer_faq,
}
