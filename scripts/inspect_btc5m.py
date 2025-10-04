import pandas as pd
df = pd.read_csv("data/cleaned/ohlcv_BTCUSD@5m.csv", parse_dates=["date"])
print("Rows:", len(df))
print("First date:", df["date"].min())
print("Last date:", df["date"].max())
