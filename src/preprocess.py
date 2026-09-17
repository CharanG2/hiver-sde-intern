# @'
# """
# Extract AppleSupport customer->company conversation pairs from twcs.csv.
# Writes: data/processed/apple_conversations.parquet
# """
import re
import pandas as pd
from src.utils import get_paths

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")

COMPANY_ID = "AppleSupport"


def clean(text, drop_mentions=True):
    if pd.isna(text):
        return ""
    t = str(text)
    t = URL_RE.sub("__url__", t)
    if drop_mentions:
        t = MENTION_RE.sub("", t)
    t = " ".join(t.split())
    return t.strip()


def extract_pairs(df):
    """
    For every AppleSupport reply, find the inbound tweet it responded to.
    That inbound tweet is the customer message.
    """
    # All AppleSupport replies that are responses to something
    ap = df[(df["author_id"] == COMPANY_ID) & df["in_response_to_tweet_id"].notna()].copy()

    # Build lookup: tweet_id -> (author, text, inbound)
    lookup = df.set_index("tweet_id")[["author_id", "text", "inbound"]].to_dict("index")

    rows = []
    for _, r in ap.iterrows():
        target_id = r["in_response_to_tweet_id"]
        target = lookup.get(target_id)
        if target is None:
            continue
        # Only keep pairs where the target is an actual customer message
        if target["inbound"] != "True":
            continue
        rows.append({
            "customer_message": target["text"],
            "company_reply": r["text"],
            "timestamp": r["created_at"],
        })

    return pd.DataFrame(rows)


def main():
    paths = get_paths()
    raw = paths["raw"] / "twcs.csv"
    if not raw.exists():
        raise FileNotFoundError(f"Missing {raw}")

    print("Loading twcs.csv (~30s)...")
    df = pd.read_csv(raw, dtype=str)
    print(f"Loaded {len(df):,} tweets")

    print("Extracting AppleSupport pairs...")
    conv = extract_pairs(df)
    print(f"Extracted {len(conv):,} raw pairs")

    conv["customer_message_clean"] = conv["customer_message"].apply(lambda x: clean(x, True))
    conv["company_reply_clean"] = conv["company_reply"].apply(lambda x: clean(x, False))

    before = len(conv)
    conv = conv[
        (conv["customer_message_clean"].str.len() > 15) &
        (conv["company_reply_clean"].str.len() > 20)
    ]
    conv = conv.drop_duplicates(subset=["customer_message_clean"]).reset_index(drop=True)
    print(f"After cleaning/dedup: {len(conv):,} (dropped {before - len(conv):,})")

    out_path = paths["processed"] / "apple_conversations.parquet"
    conv.to_parquet(out_path)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
# '@ | Out-File -Encoding utf8 src\preprocess.py