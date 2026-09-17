"""Escalation decision logic."""
from dataclasses import dataclass
from src.intent_classifier import ESCALATION_REQUIRED

SENSITIVE = ["lawyer", "sue", "legal", "attorney", "court", "safety",
             "danger", "injury", "fraud", "unauthorized", "hacked", "stolen"]


@dataclass
class Decision:
    should_escalate: bool
    reason: str


def decide(message, intent, confidence=1.0, threshold=0.7):
    # Intent-based escalation (billing, warranty, privacy, complaints)
    if ESCALATION_REQUIRED.get(intent, False):
        return Decision(True, f"Intent '{intent}' requires human handling")

    # Sensitive keyword check — always escalates
    low = (message or "").lower()
    hits = [kw for kw in SENSITIVE if kw in low]
    if hits:
        return Decision(True, f"Sensitive keywords: {', '.join(hits)}")

    # Low confidence
    if confidence < threshold:
        return Decision(True, f"Low confidence ({confidence:.2f} < {threshold})")

    return Decision(False, "Standard intent, high confidence — eligible for auto-handling")