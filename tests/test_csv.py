import pandas as pd
df = pd.read_csv("data/processed/mongo_lstm_output.csv", parse_dates=["timestamp"])
print(df.shape)
print(df["error"].describe())
print("anomalies:", df["anomaly"].sum())
df[df["anomaly"]==1].head()
