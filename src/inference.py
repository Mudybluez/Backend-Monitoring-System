import torch
import pandas as pd
import joblib
from pymongo import MongoClient
from src.model import Autoencoder
from src.config import *
import numpy as np



def load_from_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["monitoring"]
    collection = db["server_metrics"]

    docs = list(collection.find({}, {"_id": 0}))

    df = pd.DataFrame(docs)
    print(f"📌 Загружено строк из MongoDB: {len(df)}")
    return df


def preprocess_mongo(df):
    # авто-обнаружение колонок
    def pick(cols, *keywords):
        for kw in keywords:
            for col in cols:
                if kw in col.lower():
                    return col
        return None

    cols = df.columns

    cpu_col = pick(cols, "cpu")
    mem_col = pick(cols, "mem")
    disk_col = pick(cols, "disk")
    net_col = pick(cols, "net")

    df_norm = pd.DataFrame({
        "cpu": df[cpu_col],
        "memory": df[mem_col],
        "disk": df[disk_col],
        "network": df[net_col],
    })

    return df_norm.fillna(df_norm.mean())


def infer():
    print("📌 Загружаем модель...")

    scaler = joblib.load("models/checkpoints/scaler.pkl")

    model = Autoencoder(4)
    model.load_state_dict(torch.load("models/checkpoints/autoencoder.pt"))
    model.eval()

    df = load_from_mongo()
    df_norm = preprocess_mongo(df)

    X = scaler.transform(df_norm)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor).numpy()

    # reconstruction error
    errors = np.mean((X - reconstructed) ** 2, axis=1)
    df["reconstruction_error"] = errors

    # threshold
    threshold = np.percentile(errors, 98)   # top 2% = аномалии
    df["anomaly"] = df["reconstruction_error"] > threshold

    print("\n🔥 ТОП АНОМАЛИЙ:")
    print(df[df["anomaly"]].sort_values("reconstruction_error", ascending=False).head(20))

    df.to_csv("data/processed/mongo_analyzed.csv", index=False)
    print("\n📌 Результат сохранён: data/processed/mongo_analyzed.csv")


if __name__ == "__main__":
    infer()
