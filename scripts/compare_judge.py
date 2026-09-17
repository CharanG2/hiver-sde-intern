"""After you've scored the 30 replies, compute judge-vs-human agreement."""
import json
import pandas as pd
from src.utils import get_paths
from evaluation.llm_judge import judge_reply
from evaluation.judge_validation import agreement


def main():
    paths = get_paths()
    df = pd.read_csv(paths["results"] / "judge_validation_sample.csv").fillna("")
    df = df[df["human_score_1to5"].astype(str).str.len() > 0]

    if len(df) == 0:
        print("No human scores found. Fill judge_validation_sample.csv first.")
        return

    human, judge = [], []
    for _, row in df.iterrows():
        try:
            j = judge_reply(row["customer_message"], row["generated_reply"], "n/a")
            if "error" in j:
                continue
            human.append(float(row["human_score_1to5"]))
            judge.append(float(j["mean_score"]))
        except Exception as e:
            print(f"Row {row['id']} failed: {e}")

    result = agreement(human, judge)
    out = paths["results"] / "judge_validation.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()