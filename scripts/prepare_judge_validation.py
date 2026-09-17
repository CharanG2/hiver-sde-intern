"""Pick 30 auto-handled replies and give them to you to score by hand."""
import pandas as pd
from src.utils import get_paths


def main():
    paths = get_paths()
    pred = pd.read_csv(paths["results"] / "predictions.csv")
    auto = pred[pred["pred_escalate"] == False].head(30).copy()
    auto["human_score_1to5"] = ""
    out = paths["results"] / "judge_validation_sample.csv"
    auto[["id", "customer_message", "generated_reply", "human_score_1to5"]].to_csv(out, index=False)
    print(f"Wrote {out} with {len(auto)} rows.")
    print("Open it in VS Code, fill human_score_1to5 for each reply, save.")


if __name__ == "__main__":
    main()