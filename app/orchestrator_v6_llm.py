# app/orchestrator_v6_llm.py
from __future__ import annotations
from typing import Dict, Any, List
from openai import OpenAI
import json, re, os

from app.agents import SearcherAgent, PriceAgent, ReviewAgent, FAQAgent, BaseAgent

ALLOWED_STEPS = {"search", "price", "reviews", "faq"}

ROUTER_SYSTEM = (
    "You are a routing planner for a shopping assistant. "
    "Given a user query, return ONLY a JSON array of steps to run in order. "
    "Allowed steps: 'search', 'price', 'reviews', 'faq'. "
    "Rules: keep minimal and relevant; if asking for prices or reviews you may include 'search' first; "
    "do not add explanations or code fences—just the JSON array."
)

def _extract_json_array(text: str) -> List[str] | None:
    """Robustly parse a JSON array even if the model wraps it in prose or code fences."""
    # try raw first
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else None
    except Exception:
        pass
    # strip code fences
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, list) else None
    except Exception:
        return None

class LLMOrchestrator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        # Fail fast if key missing (helps Streamlit show a friendly error)
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.client = OpenAI()
        self.model = model
        self.search = SearcherAgent()
        self.pricer = PriceAgent()
        self.reviewer = ReviewAgent()
        self.faq = FAQAgent()

    def plan(self, query: str) -> List[str]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        steps = _extract_json_array(text) or ["search"]
        steps = [s for s in steps if s in ALLOWED_STEPS]
        if not steps:
            steps = ["search"]
        # Ensure we have search first if plan requests price/reviews without a prior search
        if any(s in steps for s in ("price", "reviews")) and "search" not in steps:
            steps.insert(0, "search")
        return steps

    def run(self, query: str) -> Dict[str, Any]:
        steps_out: List[Dict[str, Any]] = []
        budget = BaseAgent.parse_budget(query)
        search_query = query + (f" (budget <= {int(budget)})" if budget else "")

        plan = self.plan(query)
        first_pid = None

        for step in plan:
            if step == "search":
                s = self.search.run(search_query, top_k=5)
                steps_out.append({"step": "search", "input": {"query": search_query, "top_k": 5}, "output": s})
                if s.get("items"):
                    first_pid = first_pid or s["items"][0]["id"]
            elif step == "price":
                pid = first_pid or "p1"
                p = self.pricer.run(pid)
                steps_out.append({"step": "price", "input": {"product_id": pid}, "output": p})
            elif step == "reviews":
                pid = first_pid or "p1"
                r = self.reviewer.run(pid, top_k=3)
                steps_out.append({"step": "reviews", "input": {"product_id": pid, "top_k": 3}, "output": r})
            elif step == "faq":
                f = self.faq.run(query)
                steps_out.append({"step": "faq", "input": {"question": query}, "output": f})

        return {"plan": plan, "steps": steps_out}
