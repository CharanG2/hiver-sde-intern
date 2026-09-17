"""LLM-as-judge for reply quality. Uses Groq."""
import json
import re
from src.utils import groq_complete

RUBRIC = """Evaluate this Apple Support reply on 5 criteria, each 1-5:

1. Helpfulness: Does it address the customer's issue? (5=clear actionable solution, 1=generic)
2. Tone: Professional, empathetic? (5=warm+human, 1=dismissive)
3. Accuracy: Factually correct, no invented policies? (5=correct, 1=wrong)
4. Clarity: Easy to understand? (5=clear+concise, 1=confusing)
5. Completeness: Covers the main ask? (5=fully, 1=leaves key questions)

Return ONLY valid JSON, no other text:
{
  "helpfulness": <1-5>,
  "tone": <1-5>,
  "accuracy": <1-5>,
  "clarity": <1-5>,
  "completeness": <1-5>,
  "justification": "<one sentence>"
}
"""


def judge_reply(customer_message, generated_reply, intent):
    prompt = f"""{RUBRIC}

Customer message: {customer_message}
Classified intent: {intent}
Reply to evaluate: {generated_reply}

JSON:"""
    raw = groq_complete(prompt, temperature=0.0, max_tokens=300)
    # Extract JSON
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"error": "no json", "raw": raw}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"error": "bad json", "raw": raw}

    scores = [data.get(k, 0) for k in
              ["helpfulness", "tone", "accuracy", "clarity", "completeness"]]
    data["mean_score"] = sum(scores) / len(scores)
    return data 