# ui_v1.py
# V1 – The Simplest LLM Wrapper
# Direct prompt → LLM → response. No retrieval, no guardrails, no observability, no agents.

import os
import json
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="V1 – Simplest LLM Wrapper", page_icon="🧪", layout="centered")

st.title("V1 — Simplest LLM Wrapper")
st.caption("Direct prompt → LLM → response. No grounding. No checks. No structure enforcement.")

# --- Hardcoded demo SKUs (includes SOAP-100 per the numeric example)
SKUS = [
    {
        "sku": "SOAP-100",
        "name": "Bath Soap 100g",
        "sales_last_3_months": [60, 55, 65],  # Forecast ≈ 60 by average
        "on_hand": 80,
        "safety_stock": 40,
        "moq": 24
    },
    {
        "sku": "SHAM-250",
        "name": "Shampoo 250ml",
        "sales_last_3_months": [30, 35, 28],
        "on_hand": 20,
        "safety_stock": 25,
        "moq": 12
    },
    {
        "sku": "PASTE-75",
        "name": "Toothpaste 75g",
        "sales_last_3_months": [80, 78, 82],
        "on_hand": 50,
        "safety_stock": 50,
        "moq": 24
    },
]

with st.expander("Hardcoded dataset (V1 demo)", expanded=True):
    st.json(SKUS, expanded=False)

st.markdown("**Task**: Ask the model for a next-month forecast and a JSON order plan.")
st.write(
    "Rule of thumb passed in the prompt (not enforced by code):\n"
    "- Forecast ≈ average of last three months' sales\n"
    "- Need = max(0, forecast + safety_stock − on_hand)\n"
    "- Apply MOQ = round up to nearest multiple of `moq`\n\n"
    "_V1 limitation:_ The app just asks nicely; it doesn’t verify or correct the model."
)

# Optional free-form note to model (V1 simply forwards this)
user_note = st.text_area(
    "Optional note to the model (V1 just forwards this):",
    placeholder="E.g., Ensure SOAP-100 appears in the plan and return pure JSON.",
)

if st.button("Generate Order Plan (LLM)"):
    # --- API key guard
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.error("OPENAI_API_KEY not found in environment. Please set it and retry.")
        st.stop()

    client = OpenAI(api_key=api_key)

    # --- Messages (what we *actually* send to the model)
    system_msg = (
        "You are a precise supply planner. "
        "Given a small list of SKUs with recent sales, on_hand, safety_stock, and moq, "
        "estimate next-month forecast per SKU as the average of the last three months' sales. "
        "Then compute order quantity using: need = max(0, forecast + safety_stock - on_hand). "
        "Apply MOQ by rounding up to the nearest multiple of `moq`.\n\n"
        "Return a PURE JSON object ONLY (no markdown, no commentary) with the shape:\n"
        "{\n"
        '  \"plan\": [\n'
        '    {\"sku\": \"STRING\", \"name\": \"STRING\", \"forecast_units\": NUMBER, \"order_units\": NUMBER}\n'
        "  ]\n"
        "}\n"
    )

    user_msg = {
        "role": "user",
        "content": (
            "Here are the SKUs as JSON:\n"
            f"{json.dumps(SKUS)}\n\n"
            "Please compute the plan. " +
            (f"Note: {user_note}" if user_note else "")
        ),
    }

    # --- Show input & output in two tabs for clarity
    tab_in, tab_out = st.tabs(["Prompt (System + User)", "Output"])

    with tab_in:
        st.markdown("**System message**")
        st.code(system_msg)
        st.markdown("**User message**")
        st.code(user_msg["content"])
        st.markdown("**Call params**")
        st.json({"model": "gpt-4o-mini", "temperature": 0.2, "max_tokens": 800})

    # --- LLM call (V1: minimal; no schema enforcement)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "system", "content": system_msg}, user_msg],
            max_tokens=800,
        )
        text = resp.choices[0].message.content
    except Exception as e:
        text = f"<< LLM call failed in V1 >> {e}"

    with tab_out:
        st.subheader("Raw model output")
        st.code(text, language="json")
        # Best-effort parse (still V1; okay if it fails)
        try:
            parsed = json.loads(text)
            st.subheader("Parsed JSON (best effort)")
            st.json(parsed)
        except Exception:
            st.info("Could not parse clean JSON — expected in V1 baseline.")

    st.markdown("---")
    st.caption(
        "Teaching point (V1): Baseline first. Output may hallucinate and may not parse. "
        "V2 adds retrieval/grounding; V3 adds guardrails; V4 adds observability; V5 adds agents; V6 adds planner/self-reflect."
    )

