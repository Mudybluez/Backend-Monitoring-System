import pandas as pd
import numpy as np
import torch
import joblib
from torch import nn
from src.model import Autoencoder

MODEL_PATH = "models/checkpoints/autoencoder.pt"
SCALER_PATH = "models/checkpoints/scaler.pkl"
OUTPUT_PATH = "data/processed/anomaly_results.csv"


def load_model(input_dim):
    model = Autoencoder(input_dim=input_dim)
    state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state)
    model.eval()
    return model


def load_scaler():
    return joblib.load(SCALER_PATH)


def run_inference(input_csv):
    df = pd.read_csv(input_csv)
    print("📌 Загружен CSV:", df.shape)

    # --- Detect real column names ---
    def pick(name):
        for c in df.columns:
            if name.lower() in c.lower():
                return c
        return None

    cpu_col = pick("cpu")
    mem_col = pick("mem")
    disk_col = pick("disk")
    net_col = pick("net")

    metric_cols = [cpu_col, mem_col, disk_col, net_col]

    if None in metric_cols:
        raise ValueError("❌ Не найдены нужные колонки в CSV!")

    # --- Extract feature matrix ---
    X = df[metric_cols].astype(float)

    # --- Rename to match training columns ---
    rename = {
        cpu_col: "cpu",
        mem_col: "memory",
        disk_col: "disk",
        net_col: "network"
    }
    X = X.rename(columns=rename)

    # --- Load scaler ---
    scaler = load_scaler()
    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # --- Load model ---
    input_dim = X_tensor.shape[1]
    model = load_model(input_dim)

    # --- Inference ---
    with torch.no_grad():
        reconstructed = model(X_tensor)

    errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

    df["reconstruction_error"] = errors

    # Adaptive threshold
    threshold = np.percentile(errors, 98)
    df["anomaly_flag"] = (errors > threshold).astype(int)

    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Инференс завершён.")
    print("📁 Результаты сохранены в:", OUTPUT_PATH)


if __name__ == "__main__":
    run_inference("data/processed/mongo_export.csv")
