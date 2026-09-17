import pandas as pd
df = pd.read_csv("data/raw/twcs.csv", dtype=str)
print("columns:", list(df.columns))
print()

# All tweets authored by AppleSupport
ap = df[df["author_id"] == "AppleSupport"]
print(f"AppleSupport tweets: {len(ap):,}")
print(f"  inbound values: {ap['inbound'].value_counts().to_dict()}")
print(f"  in_response_to_tweet_id non-null: {ap['in_response_to_tweet_id'].notna().sum():,}")
print(f"  in_response_to_tweet_id null: {ap['in_response_to_tweet_id'].isna().sum():,}")
print()

# How many distinct customer tweets are they replying to?
resp_ids = ap["in_response_to_tweet_id"].dropna().unique()
print(f"Distinct tweets AppleSupport replied to: {len(resp_ids):,}")

# Sample 5 AppleSupport replies with their target tweet
print("\n--- 5 sample reply pairs ---")
sample = ap[ap["in_response_to_tweet_id"].notna()].head(5)
for _, row in sample.iterrows():
    target_id = row["in_response_to_tweet_id"]
    target = df[df["tweet_id"] == target_id]
    print(f"\nAppleSupport reply: {row['text'][:100]}")
    if len(target):
        t = target.iloc[0]
        print(f"  <- Replied to (author={t['author_id']}, inbound={t['inbound']}): {t['text'][:120]}")
    else:
        print(f"  <- Target tweet_id {target_id} NOT FOUND in dataset")

# How many of those targets are inbound=True (customer messages)?
targets = df[df["tweet_id"].isin(resp_ids)]
print(f"\nOf tweets AppleSupport replied to: {len(targets):,} found")
print(f"  inbound=True (customer): {(targets['inbound'] == 'True').sum():,}")
print(f"  inbound=False (brand):   {(targets['inbound'] == 'False').sum():,}")
