import pandas as pd
df = pd.read_csv("data/golden_set/golden_set.csv")
print(f"Rows: {len(df)}")
print()
for i, row in df.head(8).iterrows():
    print(f"--- [{i}] ---")
    print(f"Customer: {str(row['customer_message'])[:180]}")
    print(f"AppleSup: {str(row['historical_reply'])[:180]}")
    print()
