# Golden Set — Sampling and Labeling Notes

## Sampling

**Source:** 101,697 AppleSupport conversations extracted from Kaggle `customer-support-on-twitter`

**Method:** Random sample of 200 with `random_state=42`, no stratification.

**Sampled:** 200
**Labeled:** 191
**Skipped:** 9 (too fragmentary — e.g. emoji-only, single-word replies)

## Labeling Rules

### Intent

| Intent | Rules |
|---|---|
| `device_issue` | Battery drain, won't charge, screen damage, hardware freeze, overheat |
| `account_access` | Apple ID, password, 2FA, locked out, sign-in failure |
| `billing_refund` | Refunds, unauthorized charges, subscriptions, App Store money |
| `software_update` | iOS/macOS update bugs, autocorrect bug, non-hardware crashes |
| `order_delivery` | Shipping status, tracking, delivery delays |
| `warranty_repair` | Repairs, AppleCare, Genius Bar |
| `data_privacy` | Data deletion, GDPR, privacy requests |
| `general_inquiry` | "How do I…" questions, feature requests |
| `complaint_escalation` | Legal threats, requests for supervisor, extreme anger, "sue" |
| `out_of_scope` | Fragments, "thanks!", "yes", unclear, off-topic |

### Escalation

**Escalate = True** if any of:
- Intent is `billing_refund`, `warranty_repair`, `data_privacy`, or `complaint_escalation`
- Message contains legal/safety keywords: sue, lawyer, attorney, stolen, fraud, hacked
- Message is a fragment too ambiguous to handle (`out_of_scope`)

**Escalate = False** otherwise.

### Historical reply quality (1–5)

- **5** — Actually solved the problem with specific steps/version/link
- **4** — Gave useful partial help
- **3** — Acknowledged the issue but was vague
- **2** — Barely helpful ("DM us" with nothing else)
- **1** — Useless ("You're welcome!")

## Edge Cases Handled

1. **Fragments** — "Thanks!" or "Yes, I did that" — labeled `out_of_scope`, escalate=True, with note "fragment of multi-turn thread"

2. **Non-English messages** — Spanish, Portuguese, French tweets. Kept the true topic intent (e.g. `device_issue`), noted the language.

3. **Sarcasm** — "Great job on the new update! MY PHONE KEEPS FREEZING" — labeled `complaint_escalation`, not `software_update`.

4. **Multi-issue messages** — labeled with the primary pain point. Notes documented the secondary issue.

5. **Phishing / fraud reports** — "Someone paid $400 for a yoga subscription on my account and asked for my SSN" — labeled `account_access` but escalated=True with note "phishing + fraud".

## What I Didn't Do

- I did not use an LLM to generate labels. Every one of the 191 labels is my own judgment.
- I did not relabel after seeing model predictions — this would contaminate the eval set.
- I did not remove hard examples to inflate accuracy.

## Statistics

| Intent | Count |
|---|---|
| software_update | 65 |
| device_issue | 55 |
| complaint_escalation | 23 |
| out_of_scope | 21 |
| general_inquiry | 18 |
| account_access | 4 |
| warranty_repair | 2 |
| data_privacy | 1 |
| order_delivery | 1 |
| billing_refund | 1 |

**Class imbalance is real and matters.** 63% of the set is software_update + device_issue.