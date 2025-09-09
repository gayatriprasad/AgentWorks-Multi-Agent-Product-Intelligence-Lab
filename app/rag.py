from typing import List
from app.models import Recommendation
from app.store import get_product, get_reviews
from app.llm import LLM

SUM_SYS = "You summarize user reviews for products succinctly. Return JSON: {summary, pros:[], cons:[]}."

def summarize_reviews_json(llm: LLM, product_id: str) -> dict:
    reviews = get_reviews(product_id)
    blob = "\n".join([f"- ({r.rating or 'NR'}) {r.text}" for r in reviews[:25]]) or "No reviews."
    prompt = f"Summarize these reviews:\n{blob}\nReturn JSON."
    return llm.generate_json(SUM_SYS, prompt) or {"summary":"", "pros":[], "cons":[]}

def to_recommendation(product_id: str, score: float, llm: LLM) -> Recommendation:
    p = get_product(product_id)
    if not p:
        return Recommendation(product=None, score=score)
    js = summarize_reviews_json(llm, product_id)
    return Recommendation(product=p, score=score,
                          summary=js.get("summary"),
                          pros=js.get("pros") or [],
                          cons=js.get("cons") or [])

class RAGPipeline:
    def __init__(self, retriever, llm: LLM):
        self.retriever = retriever
        self.llm = llm

    def recommend(self, query: str, top_k: int = 5) -> List[Recommendation]:
        hits = self.retriever.search(query, top_k=top_k)
        return [to_recommendation(pid, score, self.llm) for pid, score in hits]
