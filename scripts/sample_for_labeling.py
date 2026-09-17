"""Sample 200 conversations and produce a CSV for hand-labeling."""
import pandas as pd
from src.utils import get_paths

def main():
    paths = get_paths()
    df = pd.read_parquet(paths["processed"] / "apple_conversations.parquet")

    sample = df.sample(n=200, random_state=42).reset_index(drop=True)
    out = pd.DataFrame({
        "id": range(len(sample)),
        "customer_message": sample["customer_message_clean"],
        "historical_reply": sample["company_reply_clean"],
        "labeled_intent": "",
        "should_escalate": "",
        "reply_quality_1to5": "",
        "notes": "",
    })
    out_path = paths["golden"] / "golden_set.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print("Next: open the CSV (or run the Streamlit app) and fill in labels.")

if __name__ == "__main__":
    main()
