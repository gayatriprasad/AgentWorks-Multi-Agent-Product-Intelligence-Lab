import os
import json
import time
from typing import List
from dataclasses import dataclass

from pydantic import BaseModel, conint, ValidationError

try:
    # OpenAI Python SDK >= 1.0 (Responses API)
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

MODEL_DEFAULT = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ---------------------------
# 1) Minimal Pydantic schema
# ---------------------------
class Order(BaseModel):
    sku: str
    qty: conint(ge=0)
    reason: str

class Plan(BaseModel):
    orders: List[Order]
    assumptions: List[str]

# ---------------------------
# 2) Hardcoded demo data (V1)
# ---------------------------
PRODUCTS = [
    {"sku": "SOAP-100", "name": "Soap 100g", "category": "Home Care", "unit_cost": 20, "moq": 24, "lead_time_days": 14},
    {"sku": "TOOTH-PASTE-1", "name": "Toothpaste 100ml", "category": "Personal Care", "unit_cost": 55, "moq": 12, "lead_time_days": 21},
    {"sku": "BRUSH-STD", "name": "Toothbrush Standard", "category": "Personal Care", "unit_cost": 18, "moq": 24, "lead_time_days": 21},
    {"sku": "SHAVE-CRM-1", "name": "Shaving Cream", "category": "Personal Care", "unit_cost": 60, "moq": 6, "lead_time_days": 21},
    {"sku": "SOAP-200", "name": "Soap 200g", "category": "Home Care", "unit_cost": 30, "moq": 24, "lead_time_days": 14},
    {"sku": "MOUTH-RINSE-1", "name": "Mouth Rinse", "category": "Personal Care", "unit_cost": 75, "moq": 6, "lead_time_days": 21},
    {"sku": "FLOSS-STD", "name": "Dental Floss", "category": "Personal Care", "unit_cost": 25, "moq": 12, "lead_time_days": 21},
    {"sku": "AFTERSHAVE-1", "name": "Aftershave", "category": "Personal Care", "unit_cost": 95, "moq": 6, "lead_time_days": 21},
    {"sku": "LOTION-1", "name": "Body Lotion", "category": "Personal Care", "unit_cost": 120, "moq": 6, "lead_time_days": 21},
    {"sku": "DEO-1", "name": "Deodorant", "category": "Personal Care", "unit_cost": 85, "moq": 6, "lead_time_days": 21},
]

# last 3 months sales per SKU (qty)
LAST_3M_SALES = {
    "SOAP-100": [60, 55, 65],
    "TOOTH-PASTE-1": [40, 42, 38],
    "BRUSH-STD": [48, 52, 51],
    "SHAVE-CRM-1": [18, 22, 19],
    "SOAP-200": [25, 27, 26],
    "MOUTH-RINSE-1": [12, 13, 12],
    "FLOSS-STD": [20, 18, 19],
    "AFTERSHAVE-1": [10, 9, 11],
    "LOTION-1": [14, 15, 14],
    "DEO-1": [22, 24, 23],
}

# on-hand inventory per SKU
ON_HAND = {
    "SOAP-100": 80,
    "TOOTH-PASTE-1": 35,
    "BRUSH-STD": 50,
    "SHAVE-CRM-1": 22,
    "SOAP-200": 20,
    "MOUTH-RINSE-1": 10,
    "FLOSS-STD": 30,
    "AFTERSHAVE-1": 12,
    "LOTION-1": 16,
    "DEO-1": 28,
}

# simple policy knobs
BUDGET = 75000
SAFETY_STOCK = {
    # minimal safety stock per SKU for demo
    sku: 40 if sku.startswith("SOAP") else 20 for sku in [p["sku"] for p in PRODUCTS]
}

# ---------------------------
# 3) Prompt builder
# ---------------------------
SYSTEM_PROMPT = (
    "You are a careful retail demand planner. "
    "Given small-store sales and inventory, forecast next-cycle demand and propose an order plan. "
    "Return ONLY a JSON object matching the schema: {\"orders\":[{\"sku\":str,\"qty\":int,\"reason\":str}],\"assumptions\":[str]}. "
    "Rules: qty must be non-negative integers; follow MOQ per SKU; do not exceed total budget; explain rationale briefly per SKU."
)

USER_TEMPLATE = """
SKUs:
{skus}

Last 3 months sales per SKU:
{sales}

On-hand inventory per SKU:
{on_hand}

Policy/Constraints:
- Order horizon: next month
- Safety stock (default): {safety_note}
- MOQ per SKU: use products.moq
- Lead time days per SKU: products.lead_time_days (use to reason, not to simulate)
- Budget (INR): {budget}

Return JSON with orders for ALL listed SKUs (even if qty=0) and assumptions. Keep reasons < 20 words each.
"""


def build_user_prompt() -> str:
    skus = [{k: v for k, v in p.items() if k in ("sku", "name", "unit_cost", "moq", "lead_time_days") } for p in PRODUCTS]
    payload = USER_TEMPLATE.format(
        skus=json.dumps(skus, indent=2),
        sales=json.dumps(LAST_3M_SALES, indent=2),
        on_hand=json.dumps(ON_HAND, indent=2),
        safety_note="40 for SOAP*, else 20",
        budget=BUDGET,
    )
    return payload

# ---------------------------
# 4) LLM call + repair loop
# ---------------------------


def call_llm(system_prompt: str, user_prompt: str, model: str = MODEL_DEFAULT) -> str:
    client = OpenAI()
    try:
        # Preferred path: JSON mode via Chat Completions
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except TypeError:
        # Fallback for older SDKs/models that don't support response_format
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nReturn ONLY a valid JSON object."},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content


def validate_or_repair(raw_text: str, max_repairs: int = 2) -> Plan:
    """Try to parse + validate JSON to Plan. If fails, ask the model to repair."""
    attempt = 0
    last_err = None
    content = raw_text

    while attempt <= max_repairs:
        try:
            data = json.loads(content)
            plan = Plan(**data)
            return plan
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = str(e)
            attempt += 1
            if attempt > max_repairs:
                break
            # ask model to repair using the error message as instruction
            repair_prompt = (
                r'You produced invalid JSON for the schema {"orders":[{"sku":str,"qty":int,"reason":str}],"assumptions":[str]}. '
                f'Fix it and return ONLY valid JSON. Error was: {last_err}'
            )

            content = call_llm(SYSTEM_PROMPT, repair_prompt)
            time.sleep(0.2)

    raise ValueError(f"Failed to validate/repair model output after {max_repairs} attempts. Last error: {last_err}")

# ---------------------------
# 5) Demo runner
# ---------------------------

def main():
    user_prompt = build_user_prompt()
    raw = call_llm(SYSTEM_PROMPT, user_prompt)
    plan = validate_or_repair(raw)

    # Print compact table + JSON
    print("\n=== Recommended Order Plan (V1) ===")
    total_cost = 0
    moq_lookup = {p["sku"]: p["moq"] for p in PRODUCTS}
    price_lookup = {p["sku"]: p["unit_cost"] for p in PRODUCTS}

    for o in plan.orders:
        cost = o.qty * price_lookup.get(o.sku, 0)
        total_cost += cost
        moq = moq_lookup.get(o.sku, 1)
        print(f"{o.sku:15} qty={o.qty:4d}  moq={moq:3d}  cost={cost:7.0f}  reason={o.reason}")

    print(f"Total cost (INR): {total_cost}")
    if total_cost > BUDGET:
        print("WARNING: Proposed plan exceeds budget. (V1 does not auto-adjust; later versions will.)")

    print("\n--- Raw JSON ---")
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    # Expect environment variable OPENAI_API_KEY to be set.
    if os.getenv("OPENAI_API_KEY") is None:
        print("[ERROR] Please set OPENAI_API_KEY env var.")
        raise SystemExit(1)
    main()