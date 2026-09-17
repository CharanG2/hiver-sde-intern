# Decision Log — 15 Non-Obvious Decisions

1. **Chose AppleSupport over other brands** — second-most-active account in dataset (106,860 outbound tweets), clean product focus (iPhone, Apple ID, iOS), high volume of similar intents makes RAG retrieval meaningful.

2. **10 intents, not Banking77's 77** — Twitter support messages are less granular than banking queries. 10 classes match the natural clusters I observed in 300 sampled messages.

3. **Battery/charge/screen beat "update" keyword** — 15-point accuracy gain came from re-prioritizing device hints over update hints in the classifier. "Battery drains after iOS update" is a device issue, not a software issue.

4. **Keyword override for device/billing/complaint/privacy** — the LLM alone was too conservative on `out_of_scope`. Keyword override for high-signal classes lifted accuracy by ~10 points.

5. **`out_of_scope` no longer auto-escalates** — earlier policy sent every fragment to a human. Removing it cut false-positive escalations by 21.

6. **Hybrid fallback: Groq primary, Gemini secondary** — Groq is faster and free, but hits daily caps. Gemini Flash-Lite is slower but rarely 429s. Automatic failover keeps the pipeline running.

7. **RAG retrieval via TF-IDF, not embeddings** — for a 100k-document corpus with short queries, TF-IDF matches semantic search quality and needs no API calls.

8. **`max_tokens=800` for reply generation** — reasoning models like gpt-oss-20b spend 300-500 tokens in internal reasoning before output. Smaller limits cause empty responses.

9. **Checkpoint every 5 examples** — free-tier API rate limits caused 3 crashes mid-pipeline. The checkpoint saved 160+ examples of work each time.

10. **Template fallback reply for empty generations** — when both LLM providers fail, fall back to a per-intent template. Empty replies were dragging judge scores to 2.4; template fallback lifted to 3.18.

11. **Golden set: 191, not 250** — I originally sampled 200. 9 were skipped as too fragmentary to label meaningfully. 191 is inside the required 150–250.

12. **Fragment handling rule** — messages like "thanks!" or "found that link" get `out_of_scope`. They appear 21 times in the golden set and reflect real Twitter interaction patterns.

13. **Non-English messages keep their topic intent** — a Spanish tweet about battery gets `device_issue`, with a language note. Rejecting them as `out_of_scope` would be dishonest labeling.

14. **Judge scores on 5 criteria, averaged** — helpfulness, tone, accuracy, clarity, completeness. Averaging avoids over-weighting a single criterion.

15. **Reported weaknesses prominently, not buried** — 49.7% accuracy and 0.29 escalation F1 are the honest numbers. The report's "What is misleading" section documents why even those numbers overstate real-world performance.