import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dataset_loader import load_all_datasets
from src.preprocess import normalize_and_merge
from src.model import Autoencoder
from src.config import *

import joblib
import os


def train():
    print("📌 Загружаем датасеты...")
    dfs = load_all_datasets()

    print("📌 Preprocess...")
    X, df_raw = normalize_and_merge(dfs)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X.shape[1]
    print(f"📌 Input dim: {input_dim}")

    device = torch.device(DEVICE)
    model = Autoencoder(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("📌 Начинаем обучение...")

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in loader:
            batch = batch[0].to(device)

            output = model(batch)
            loss = criterion(output, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}   Loss: {total_loss:.6f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{MODEL_DIR}/autoencoder.pt")

    print("\n✅ Модель сохранена!")
    print("📌 Путь:", f"{MODEL_DIR}/autoencoder.pt")

    print("📌 Сохранён scaler в models/checkpoints/scaler.pkl")

if __name__ == "__main__":
    train()
