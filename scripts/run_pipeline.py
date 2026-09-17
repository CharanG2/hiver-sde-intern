"""Run the full agent on the golden set and compute metrics. Checkpointed."""
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils import get_paths
from src.agent import AppleSupportAgent
from evaluation.metrics import intent_metrics, escalation_metrics
from evaluation.llm_judge import judge_reply


def to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _mean(items, key):
    vals = [i[key] for i in items if key in i]
    return float(sum(vals) / len(vals)) if vals else 0.0


def main():
    paths = get_paths()
    conv = pd.read_parquet(paths["processed"] / "apple_conversations.parquet")
    gold = pd.read_csv(paths["golden"] / "golden_set.csv").fillna("")
    gold = gold[gold["labeled_intent"].astype(str).str.len() > 0].reset_index(drop=True)
    print(f"Evaluating on {len(gold)} labeled examples")

    ckpt_path = paths["results"] / "checkpoint.csv"
    if ckpt_path.exists():
        ckpt = pd.read_csv(ckpt_path)
        done_ids = set(ckpt["id"].tolist())
        print(f"Resuming from checkpoint: {len(done_ids)} already done")
        outputs = ckpt.to_dict("records")
    else:
        done_ids = set()
        outputs = []

    agent = AppleSupportAgent(conv)

    for _, row in tqdm(gold.iterrows(), total=len(gold)):
        rid = int(row["id"])
        if rid in done_ids:
            continue

        out = agent.process(row["customer_message"])

        # Judge
        judge = {}
        try:
            j = judge_reply(row["customer_message"], out.reply, out.intent)
            if "error" not in j:
                judge = j
        except Exception as e:
            print(f"Judge failed on row {rid}: {e}")

        outputs.append({
            "id": rid,
            "customer_message": row["customer_message"],
            "true_intent": row["labeled_intent"],
            "pred_intent": out.intent,
            "true_escalate": to_bool(row["should_escalate"]),
            "pred_escalate": out.escalation == "escalated",
            "escalation_reason": out.escalation_reason,
            "generated_reply": out.reply,
            "judge_helpfulness": judge.get("helpfulness", ""),
            "judge_tone": judge.get("tone", ""),
            "judge_accuracy": judge.get("accuracy", ""),
            "judge_clarity": judge.get("clarity", ""),
            "judge_completeness": judge.get("completeness", ""),
            "judge_mean": judge.get("mean_score", ""),
        })

        # Save checkpoint every 5 examples
        if len(outputs) % 5 == 0:
            pd.DataFrame(outputs).to_csv(ckpt_path, index=False)

    # Final save
    df = pd.DataFrame(outputs)
    df.to_csv(ckpt_path, index=False)
    df.to_csv(paths["results"] / "predictions.csv", index=False)

    # Compute metrics
    y_true = df["true_intent"].tolist()
    y_pred = df["pred_intent"].tolist()
    esc_true = [1 if v else 0 for v in df["true_escalate"]]
    esc_pred = [1 if v else 0 for v in df["pred_escalate"]]

    judge_rows = df[df["judge_mean"] != ""].copy()
    for k in ["judge_helpfulness", "judge_tone", "judge_accuracy",
              "judge_clarity", "judge_completeness", "judge_mean"]:
        judge_rows[k] = pd.to_numeric(judge_rows[k], errors="coerce")

    results = {
        "n_examples": len(df),
        "intent": intent_metrics(y_true, y_pred),
        "escalation": escalation_metrics(esc_true, esc_pred),
        "judge": {
            "n_judged": int(len(judge_rows)),
            "mean_helpfulness": float(judge_rows["judge_helpfulness"].mean()) if len(judge_rows) else 0.0,
            "mean_tone": float(judge_rows["judge_tone"].mean()) if len(judge_rows) else 0.0,
            "mean_accuracy": float(judge_rows["judge_accuracy"].mean()) if len(judge_rows) else 0.0,
            "mean_clarity": float(judge_rows["judge_clarity"].mean()) if len(judge_rows) else 0.0,
            "mean_completeness": float(judge_rows["judge_completeness"].mean()) if len(judge_rows) else 0.0,
            "mean_overall": float(judge_rows["judge_mean"].mean()) if len(judge_rows) else 0.0,
        },
    }

    out_path = paths["results"] / "evaluation_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("\nResults:")
    print(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()