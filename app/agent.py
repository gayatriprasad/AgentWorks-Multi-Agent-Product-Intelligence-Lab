from typing import List
from app.retriever import SearchIndex
from app.llm import LLM
from app.rag import summarize_reviews_json, to_recommendation
from app.store import get_product, get_all_products
from app.query import parse_query

class RecommenderAgent:
    def __init__(self, model: str | None = None):
        self.llm = LLM(model=model)
        self.idx = SearchIndex()

    def recommend(self, query: str, top_k: int = 5):
        parsed = parse_query(query)
        candidates = self.idx.search(query, top_k=200)  # widen pool, we’ll filter

        def category_ok(p):
            if not parsed["category"]:
                return True
            if (p.category or "").lower().find(parsed["category"].lower()) != -1:
                return True
            txt = f"{p.title} {p.description or ''}".lower()
            return any(s in txt for s in ["earbuds","earbud","tws","true wireless","buds"])

        def price_ok(p):
            return (parsed["max_price"] is None or p.price is None or p.price <= parsed["max_price"])

        filtered = []
        for pid, score in candidates:
            p = get_product(pid)
            if p and category_ok(p) and price_ok(p):
                # light bonus: must-terms + price closeness (under budget)
                bonus = 0.0
                txt = f"{p.title} {p.description or ''}".lower()
                if parsed["terms"]:
                    hits = sum(1 for t in parsed["terms"] if t in txt)
                    bonus += 0.05 * hits
                if parsed["max_price"] and p.price:
                    budget = parsed["max_price"]
                    if p.price <= budget:
                        closeness = 1.0 - (budget - p.price) / budget  # closer to budget ⇒ higher
                        bonus += 0.15 * max(0.0, min(1.0, closeness))
                filtered.append((pid, score + bonus))

        # If nothing passed (rare), fall back to category-only
        if not filtered and parsed["category"]:
            for pid, score in candidates:
                p = get_product(pid)
                if p and category_ok(p):
                    filtered.append((pid, score))

        filtered.sort(key=lambda x: x[1], reverse=True)
        hits = filtered[:top_k]
        return [to_recommendation(pid, sc, self.llm) for pid, sc in hits]

    def summarize_product(self, product_id: str):
        p = get_product(product_id)
        if not p:
            return {"error": f"Product '{product_id}' not found in store."}
        return summarize_reviews_json(self.llm, product_id)
