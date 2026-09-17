"""End-to-end agent."""
from dataclasses import dataclass
from src.intent_classifier import IntentClassifier
from src.reply_generator import ReplyGenerator
from src.escalation import decide


@dataclass
class AgentOutput:
    customer_message: str
    intent: str
    reply: str                    # always generated, even if escalated
    escalation: str               # "auto_handled" or "escalated"
    escalation_reason: str
    retrieved: list


class AppleSupportAgent:
    def __init__(self, conversations):
        self.classifier = IntentClassifier()
        self.generator = ReplyGenerator(conversations)

    def process(self, message):
        intent = self.classifier.classify(message)
        examples = self.generator.retrieve(message, top_k=3)
        dec = decide(message, intent)

        # Always generate a reply (used as evidence even when escalated)
        try:
            reply = self.generator.generate(message, intent, examples)
        except Exception as e:
            reply = f"[generation failed: {e}]"

        return AgentOutput(
            customer_message=message,
            intent=intent,
            reply=reply,
            escalation="escalated" if dec.should_escalate else "auto_handled",
            escalation_reason=dec.reason,
            retrieved=examples,
        )