import pandas as pd

df = pd.read_csv("data/raw/twcs.csv", dtype=str)
print(f"Total tweets: {len(df):,}")

outbound = df[df["inbound"] == "False"]
print(f"Outbound (company) tweets: {len(outbound):,}")

top = outbound["author_id"].value_counts().head(40)
print("\nTop 40 outbound authors (likely company accounts):")
print(top)

print("\n--- Sample content per top author ---")
for author in top.head(30).index:
    sample = outbound[outbound["author_id"] == author]["text"].dropna().head(3).tolist()
    print(f"\nAuthor ID: {author}  (count={top[author]})")
    for s in sample:
        print(f"  > {str(s)[:120]}")
