"""Intent classification via Groq + TF-IDF baseline."""
import re

INTENTS = [
    ("device_issue", "Hardware problems: won't turn on, screen, battery drain, charging, freezing hardware", False),
    ("account_access", "Apple ID, password, 2FA, account locked, sign in problems", False),
    ("billing_refund", "Charges, subscriptions, refund requests, App Store money", True),
    ("software_update", "iOS/macOS update issues, crashes, apps bugs, keyboard/autocorrect bug", False),
    ("order_delivery", "Order status, shipping, delivery, tracking", False),
    ("warranty_repair", "Repairs, warranty claims, AppleCare, service", True),
    ("data_privacy", "Data deletion, privacy, account deletion", True),
    ("general_inquiry", "Product questions, feature requests, 'how do I'", False),
    ("complaint_escalation", "Anger, legal threats, requests for supervisor", True),
    ("out_of_scope", "Fragments, thanks, unclear, off-topic", False),
]

INTENT_NAMES = [i[0] for i in INTENTS]
ESCALATION_REQUIRED = {i[0]: i[2] for i in INTENTS}

EXAMPLES = """Examples:
"I updated and now my battery drains in 2 hours" -> device_issue
"My phone won't charge anymore" -> device_issue
"I was charged twice for the same app" -> billing_refund
"I can't log into my Apple ID" -> account_access
"Where is my iPhone X order?" -> order_delivery
"My screen cracked, need repair" -> warranty_repair
"Please delete my account and data" -> data_privacy
"How do I enable dark mode?" -> general_inquiry
"This is unacceptable I want to sue" -> complaint_escalation
"Thanks!" -> out_of_scope"""


def _intent_block():
    return "\n".join(f"- {name}: {desc}" for name, desc, _ in INTENTS)


PROMPT_TEMPLATE = f"""You classify Apple Support customer tweets. Pick ONE intent from the list.

Intents:
{_intent_block()}

Rules:
- Prefer the MOST SPECIFIC intent.
- Battery drain, charging, screen, hardware freezing = device_issue
- Only keyboard/autocorrect bugs and NON-hardware crashes = software_update
- Refund requests or unauthorized charges = billing_refund
- Apple ID / password / locked out = account_access
- Anger + legal threats = complaint_escalation
- Only short fragments like "thanks" or "yes" = out_of_scope

{EXAMPLES}

Now classify this message. End your answer with exactly:
ANSWER: <intent_name>

Message: {{message}}"""


# Keyword hints — strong signals that should override a model that says out_of_scope
DEVICE_HINTS = ["battery", "charg", "screen", "won't turn", "wont turn", "freez", "overheat"]
UPDATE_HINTS = ["update", "ios ", "ios1", "autocorrect", "keyboard", "question mark"]
BILLING_HINTS = ["refund", "charged", "subscription", "unauthorized charge"]
ACCESS_HINTS = ["apple id", "password", "locked", "sign in", "login", "2fa"]
ORDER_HINTS = ["order", "shipping", "delivery", "tracking"]
REPAIR_HINTS = ["repair", "warranty", "applecare", "genius bar"]
PRIVACY_HINTS = ["delete my", "privacy", "gdpr", "delete account"]
COMPLAINT_HINTS = ["sue", "lawyer", "attorney", "supervisor", "unacceptable", "worst"]


def _keyword_classify(message):
    """Pure keyword-based classifier — used as override and fallback."""
    low = (message or "").lower()

    # Priority 1: legal/safety → complaint_escalation
    if any(k in low for k in COMPLAINT_HINTS):
        return "complaint_escalation"
    # Priority 2: billing
    if any(k in low for k in BILLING_HINTS):
        return "billing_refund"
    # Priority 3: privacy
    if any(k in low for k in PRIVACY_HINTS):
        return "data_privacy"
    # Priority 4: account access
    if any(k in low for k in ACCESS_HINTS):
        return "account_access"
    # Priority 5: repair / warranty
    if any(k in low for k in REPAIR_HINTS):
        return "warranty_repair"
    # Priority 6: order / delivery
    if any(k in low for k in ORDER_HINTS):
        return "order_delivery"

    # Priority 7: DEVICE wins over UPDATE — key fix
    # Battery, charge, screen, freeze, overheat = device_issue even if "update" is mentioned
    if any(k in low for k in DEVICE_HINTS):
        return "device_issue"

    # Priority 8: pure software update (no device symptom)
    if any(k in low for k in UPDATE_HINTS):
        return "software_update"

    return None


class IntentClassifier:
    def __init__(self, model="openai/gpt-oss-20b"):
        from src.utils import groq_complete
        self._groq = groq_complete
        self.model = model

    def _parse(self, raw):
        """Extract intent from the strict ANSWER: marker."""
        if not raw:
            return None
        m = re.search(r"ANSWER:\s*([a-z_]+)", raw, re.IGNORECASE)
        if m:
            label = m.group(1).strip().lower()
            if label in INTENT_NAMES:
                return label
        return None

    def classify(self, message):
        prompt = PROMPT_TEMPLATE.format(message=message)
        raw = self._groq(prompt, model=self.model, temperature=0.0, max_tokens=500)
        label = self._parse(raw or "")
        kw = _keyword_classify(message)

        # Keywords strongly identify these — trust them over the LLM
        if kw in ("device_issue", "billing_refund", "complaint_escalation", "data_privacy"):
            return kw

        # Otherwise prefer the LLM's answer
        if label:
            return label

        if kw:
            return kw
        return "out_of_scope"


class TfidfBaseline:
    """Simple baseline for comparison."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, messages, labels):
        X = self.vec.fit_transform(messages)
        self.clf.fit(X, labels)

    def predict(self, message):
        X = self.vec.transform([message])
        return self.clf.predict(X)[0]