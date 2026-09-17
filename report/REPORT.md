# Hiver SDE Intern — AI Support Agent for AppleSupport

**Author:** Charan
**Date:** 17 September 2026
**Repo:** https://github.com/CharanG2/hiver-sde-intern

---

## 1. Problem Framing

### What "good" means for AppleSupport

For an AI agent handling AppleSupport's Twitter mentions, "good" means four things:

1. **Correctly triage** — a customer with a broken iPhone should not be routed to billing; a refund request should not be auto-answered with a device troubleshooting script.
2. **Reply grounded in history** — AppleSupport has a distinct voice (concise, warm, action-oriented). Replies should sound like the brand, not like a generic chatbot.
3. **Know when to escalate** — refunds, warranty claims, privacy requests, legal threats must go to a human. Everything else can be auto-handled.
4. **Be measurable** — without a labeled evaluation set, no claim about quality is meaningful.

### What I chose NOT to build

- **Multi-turn dialogue state** — the assignment is per-message; I treat each customer tweet as an independent unit.
- **Fine-tuned model** — RAG + prompting gives sufficient quality at zero training cost.
- **Real-time Twitter integration** — evaluated offline on historical data.
- **Sentiment analysis as a separate head** — sentiment is implicit in the intent classes (`complaint_escalation`).

---

## 2. Data and Golden Set

### Dataset

- **Source:** Kaggle `thoughtvector/customer-support-on-twitter` (~3M tweets)
- **Filtered:** AppleSupport conversations only
- **Yield:** 101,697 clean customer→AppleSupport pairs after dedup and cleaning

### Golden evaluation set

- **Sampled:** 200 random conversations (`random_state=42`) from the 101,697
- **Labeled:** 191 by hand (9 were skipped as fragments too ambiguous to label meaningfully)
- **Labels:** intent (10 classes), escalation (binary), historical reply quality (1–5)
- **Sampling rationale:** Random sampling avoids selection bias. Each example was labeled by me, not by an LLM.
- **Labeling rules:**
  - Fragments ("thanks", "yes", "found that link") → `out_of_scope`, escalate=True
  - Non-English messages → keep the true topic intent, note the language
  - Legal / safety keywords (sue, lawyer, stolen, fraud) → escalate=True regardless of intent
  - Historical reply quality: 5=solves it, 3=acknowledges, 1=pure "DM us"

See `data/golden_set/labeling_notes.md` for full rules.

---

## 3. System Architecture
