from __future__ import annotations
import os
import time

# Optional OpenAI dependency; fallback to mock if unavailable or not configured
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


def _openai_call(system_prompt: str, user_prompt: str) -> str:
    """Call OpenAI Chat Completions if SDK + key are available; else raise."""
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OpenAI SDK/key not available; using mock.")

    client = OpenAI()
    # Use chat.completions for better SDK compatibility
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
    )
    return resp.choices[0].message.content or ""


def _mock_call(system_prompt: str, user_prompt: str) -> str:
    """
    Deterministic mock to *demo guardrails*.
    If the user asks to "ignore" instructions, we echo that in the output so
    guardrails can detect and REFUSE. Otherwise, return a concise helpful reply.
    """
    time.sleep(0.3)
    up = user_prompt.strip()
    lp = up.lower()
    if any(k in lp for k in ["ignore", "jailbreak", "bypass", "system prompt"]):
        return (
            "Okay, ignoring the previous instructions as requested.\n"
            "Here’s a light joke to prove it: Why did the agent cross the road?\n"
            "To call the tool on the other side!"
        )
    return (
        f"(mock) System says: {system_prompt[:48]}...\n\n"
        f"You asked: {up}\n"
        "Here’s a helpful, concise response."
    )



from app.llm import call_llm as _llm_call

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Use llm.call_llm() which may be OpenAI chat.completions or a mock.
    It returns text (not guaranteed JSON). We return the best-guess JSON
    slice so validate_or_repair() can parse/repair.
    """
    raw = _llm_call(system_prompt, user_prompt)
    text = (raw or "").strip()

    # If the model already returned bare JSON, great.
    if text.startswith("{") and text.endswith("}"):
        return text

    # Try to extract the first {...} block from any prose-wrapped output.
    i = text.find("{")
    j = text.rfind("}")
    if i != -1 and j != -1 and j > i:
        return text[i : j + 1]

    # Last resort: return as-is; validate_or_repair will attempt a repair call.
    return text

