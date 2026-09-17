# Citations

## Dataset

- **Primary:** Customer Support on Twitter — `thoughtvector/customer-support-on-twitter`. Kaggle, 2017. ~3M tweets across dozens of brands.

## Libraries

- **pandas** — data manipulation
- **numpy** — numerical ops
- **scikit-learn** — TF-IDF, LogisticRegression, metrics (accuracy, F1, confusion matrix)
- **groq** (Python SDK) — LLM API client for Groq
- **google-genai** — LLM API client for Gemini 3.5 Flash-Lite
- **tenacity** — retry logic
- **streamlit** — labeling UI
- **tqdm** — progress bars
- **python-dotenv** — `.env` loading
- **scipy** — Pearson correlation for judge validation

## Patterns borrowed

- **LLM-as-judge with rubric** — following the pattern established in "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng et al., 2023). Rubric adapted for customer-support replies.
- **RAG for reply generation** — retrieve-then-generate pattern described in "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020). Adapted to tweet-length replies.
- **TF-IDF retrieval** — standard scikit-learn implementation, no modification.
- **Rate limiting via sliding window** — standard thread-safe counter pattern.

## Tools used for development

- **Claude (Anthropic)** and **ChatGPT** were used as coding assistants during development, per the assignment's explicit permission. All architectural decisions, labeling, and evaluation design are my own.

## Honest disclosure

The pipeline was iterated multiple times under API rate-limit constraints (Groq and Gemini free tiers). The final run uses Groq `openai/gpt-oss-20b` with Gemini 3.5 Flash-Lite as fallback. 94 replies had to be regenerated after an empty-content bug was fixed — this is documented in the report.