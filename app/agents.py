from __future__ import annotations
from typing import Dict, Any, Optional
import re
from app.tools import FUNCTION_REGISTRY

class AgentError(Exception):
    pass

class BaseAgent:
    name: str = "base"
    description: str = ""

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        fn = FUNCTION_REGISTRY.get(tool_name)
        if not fn:
            raise AgentError(f"Unknown tool: {tool_name}")
        return fn(args)

    @staticmethod
    def parse_budget(text: str) -> Optional[float]:
        m = re.search(r"(?:(?:under|below|<=?)\s*([0-9][0-9,]*)(?:\s*k)?)|([0-9][0-9,]*)\s*(?:inr|rs)\b", text, re.I)
        if not m:
            m = re.search(r"([0-9][0-9,]*)\s*(?:k)\b", text, re.I)
            if m:
                return float(m.group(1).replace(",", "")) * 1000.0
            return None
        num = m.group(1) or m.group(2)
        if not num:
            return None
        v = float(num.replace(",", ""))
        if v < 500:
            v *= 1000.0
        return v

class SearcherAgent(BaseAgent):
    name = "searcher"
    description = "Find candidate products that match constraints."
    def run(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        return self.call_tool("search_products", {"query": query, "top_k": top_k})

class PriceAgent(BaseAgent):
    name = "pricer"
    description = "Compare prices for a chosen product."
    def run(self, product_id: str) -> Dict[str, Any]:
        return self.call_tool("compare_prices", {"product_id": product_id})

class ReviewAgent(BaseAgent):
    name = "reviewer"
    description = "Fetch and summarize reviews."
    def run(self, product_id: str, top_k: int = 3) -> Dict[str, Any]:
        return self.call_tool("fetch_reviews", {"product_id": product_id, "top_k": top_k})

class FAQAgent(BaseAgent):
    name = "faq"
    description = "Answer a short question from KB."
    def run(self, question: str) -> Dict[str, Any]:
        return self.call_tool("answer_faq", {"question": question})
