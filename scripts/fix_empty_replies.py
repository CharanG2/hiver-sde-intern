"""Regenerate empty replies and re-judge them, without re-running the classifier."""
import pandas as pd
from tqdm import tqdm
from src.utils import get_paths
from src.reply_generator import ReplyGenerator
from evaluation.llm_judge import judge_reply


def main():
    paths = get_paths()
    ckpt = pd.read_csv(paths["results"] / "checkpoint.csv").fillna("")
    conv = pd.read_parquet(paths["processed"] / "apple_conversations.parquet")

    gen = ReplyGenerator(conv)
    empty_mask = ckpt["generated_reply"].astype(str).str.len() < 20
    print(f"Empty/short replies: {empty_mask.sum()}")

    fixed = 0
    for idx, row in tqdm(ckpt[empty_mask].iterrows(), total=int(empty_mask.sum())):
        try:
            examples = gen.retrieve(row["customer_message"], top_k=3)
            reply = gen.generate(row["customer_message"], row["pred_intent"], examples)
            if reply and len(reply) > 20:
                ckpt.at[idx, "generated_reply"] = reply
                try:
                    j = judge_reply(row["customer_message"], reply, row["pred_intent"])
                    if "error" not in j:
                        ckpt.at[idx, "judge_helpfulness"] = j.get("helpfulness", "")
                        ckpt.at[idx, "judge_tone"] = j.get("tone", "")
                        ckpt.at[idx, "judge_accuracy"] = j.get("accuracy", "")
                        ckpt.at[idx, "judge_clarity"] = j.get("clarity", "")
                        ckpt.at[idx, "judge_completeness"] = j.get("completeness", "")
                        ckpt.at[idx, "judge_mean"] = j.get("mean_score", "")
                except Exception as e:
                    print(f"Judge failed on row {idx}: {e}")
                fixed += 1
        except Exception as e:
            print(f"Row {idx} failed: {e}")

        if fixed > 0 and fixed % 10 == 0:
            ckpt.to_csv(paths["results"] / "checkpoint.csv", index=False)

    ckpt.to_csv(paths["results"] / "checkpoint.csv", index=False)
    print(f"\nFixed {fixed} replies.")


if __name__ == "__main__":
    main()