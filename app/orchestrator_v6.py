from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass
from app.agents import SearcherAgent, PriceAgent, ReviewAgent, FAQAgent, BaseAgent

@dataclass
class StepResult:
    agent: str
    action: str
    input: Dict[str, Any]
    output: Dict[str, Any]

class SimpleOrchestrator:
    def __init__(self):
        self.search = SearcherAgent()
        self.pricer = PriceAgent()
        self.reviewer = ReviewAgent()
        self.faq = FAQAgent()

    def route(self, query: str) -> List[str]:
        q = query.lower()
        intents: List[str] = []
        if any(k in q for k in ["price", "compare", "cost", "quote"]):
            intents.append("price")
        if any(k in q for k in ["review", "rating", "feedback"]):
            intents.append("reviews")
        if any(k in q for k in ["faq", "how to", "what is", "does it"]):
            intents.append("faq")
        if not intents:
            intents.append("search")
        return intents

    def run(self, query: str) -> Dict[str, Any]:
        steps: List[StepResult] = []
        budget = BaseAgent.parse_budget(query)
        search_query = query + (f" (budget <= {int(budget)})" if budget else "")
        s_out = self.search.run(search_query, top_k=5)
        steps.append(StepResult("searcher", "search_products",
                                {"query": search_query, "top_k": 5}, s_out))

        first_pid = None
        try:
            first_pid = s_out["items"][0]["id"] if s_out.get("items") else None
        except Exception:
            first_pid = None

        for intent in self.route(query):
            if intent == "price":
                pid = first_pid or "p1"
                p_out = self.pricer.run(pid)
                steps.append(StepResult("pricer", "compare_prices", {"product_id": pid}, p_out))
            elif intent == "reviews":
                pid = first_pid or "p1"
                r_out = self.reviewer.run(pid, top_k=3)
                steps.append(StepResult("reviewer", "fetch_reviews",
                                        {"product_id": pid, "top_k": 3}, r_out))
            elif intent == "faq":
                f_out = self.faq.run(query)
                steps.append(StepResult("faq", "answer_faq", {"question": query}, f_out))
            elif intent == "search":
                pass

        final = self.summarize(query, steps)
        return {"final": final, "trace": [s.__dict__ for s in steps]}

    def summarize(self, query: str, steps: List[StepResult]) -> str:
        lines: List[str] = [f"**Query:** {query}"]
        for s in steps:
            if s.action == "search_products":
                items = s.output.get("items", [])
                if items:
                    lines.append("**Top match:** " + items[0].get("name", "?"))
            elif s.action == "compare_prices":
                quotes = s.output.get("quotes", [])
                if quotes:
                    best = sorted(quotes, key=lambda x: x.get("price", 1e9))[0]
                    lines.append(f"**Best price:** {best['price']} {best.get('currency','INR')} @ {best['vendor']}")
            elif s.action == "fetch_reviews":
                revs = s.output.get("reviews", [])
                if revs:
                    lines.append(f"**Reviews (sample):** \"{revs[0]['text']}\"")
            elif s.action == "answer_faq":
                ans = s.output.get("answer", "")
                lines.append(f"**FAQ:** {ans}")
        return "\n".join(lines)
