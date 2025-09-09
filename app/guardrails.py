from __future__ import annotations
import re
from enum import Enum
from typing import Optional, Tuple

class Action(str, Enum):
    PASS = "pass"
    REFUSE = "refuse"
    SANITIZE = "sanitize"

# Simple markers that indicate the model is acknowledging an injection/jailbreak
_INJECTION_MARKERS = [
    "ignore previous",
    "ignore the instructions",
    "system prompt",
    "developer message",
    "bypass the guard",
    "jailbreak",
]

_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(r"\+?\d[\d\-\s]{9,}\d")  # 10+ digits (with separators)
_LONG_DIGITS_RE = re.compile(r"\d{12,}")            # Long IDs like acct/card

# Tiny profanity softener demo (swap out for a real list if needed)
PROFANITY = {
    "darn": "d**n",
}

def _mask(text: str) -> Tuple[str, Optional[str]]:
    """Mask emails, phone numbers, and very long digit strings. Light profanity softening."""
    reason_parts = []

    def _mask_email(m: re.Match) -> str:
        _local, _at, domain = m.group(0).partition("@")
        return f"***@{domain}"

    masked = _EMAIL_RE.sub(_mask_email, text)
    if masked != text:
        reason_parts.append("Masked email(s)")
    text = masked

    masked = _PHONE_RE.sub("[phone masked]", text)
    if masked != text:
        reason_parts.append("Masked phone numbers")
    text = masked

    masked = _LONG_DIGITS_RE.sub(lambda m: "*" * len(m.group(0)), text)
    if masked != text:
        reason_parts.append("Masked long numeric identifiers")
    text = masked

    # Light profanity softening
    for bad, repl in PROFANITY.items():
        if re.search(rf"(?i)\b{re.escape(bad)}\b", text):
            text = re.sub(rf"(?i)\b{re.escape(bad)}\b", repl, text)
            reason_parts.append("Softened profanity")

    reason = ", ".join(reason_parts) if reason_parts else None
    return text, reason


def apply_guardrails(model_output: str, user_prompt: str | None = None):
    lower_out = model_output.lower()
    lower_in = (user_prompt or "").lower()

    if any(m in lower_out for m in _INJECTION_MARKERS) or any(m in lower_in for m in _INJECTION_MARKERS):
        return (
            Action.REFUSE,
            "I can't comply with requests that attempt to override safety or system instructions.",
            "Prompt-injection indicator in input/output",
        )

    masked, reason = _mask(model_output)
    if reason:
        return Action.SANITIZE, masked, reason

    return Action.PASS, model_output, None
