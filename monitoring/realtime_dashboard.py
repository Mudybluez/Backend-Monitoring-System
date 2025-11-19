import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import joblib
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px

from src.model import Autoencoder

# ===========================
# CONFIG
# ===========================
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLL_NAME = os.getenv("COLLECTION_NAME")
TELEGRAM_TOKEN = os.getenv("TG_BOT_TOKEN")
TELEGRAM_CHAT = os.getenv("TG_CHAT_ID")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLL_NAME]

MODEL_PATH = "models/checkpoints/autoencoder.pt"
SCALER_PATH = "models/checkpoints/scaler.pkl"

# ===========================
# LOAD MODEL + SCALER
# ===========================
device = torch.device("cpu")

scaler = joblib.load(SCALER_PATH)
model = Autoencoder(input_dim=4)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# ===========================
# TELEGRAM ALERT
# ===========================
import requests

def send_alert(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT, "text": msg})


# ===========================
# BUILD DASHBOARD
# ===========================
st.set_page_config(layout="wide")
st.title("📡 Cloud Metrics Monitoring Dashboard")


# ===========================
# HELPER – QUERY MONGO
# ===========================
def load_latest():
    docs = list(collection.find({}, {"_id":0}))
    if len(docs) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Convert long → wide
    wide = df.pivot_table(index="timestamp", columns="metric", values="value").reset_index()

    # Standardize column names
    rename_map = {
        "CPU_Usage (%)": "cpu",
        "Memory_Usage (MB)": "memory",
        "Disk_IO (MBps)": "disk",
        "Network_Usage (MBps)": "network"
    }
    wide = wide.rename(columns=rename_map)

    return wide.dropna()


# ===========================
# PREDICT ANOMALIES
# ===========================
def detect_anomalies(w):
    X = w[["cpu", "memory", "disk", "network"]]
    X_scaled = scaler.transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(X_tensor).numpy()

    errors = np.mean((X_scaled - reconstructed)**2, axis=1)
    w["recon_error"] = errors
    w["anomaly"] = (errors > errors.mean() + 3 * errors.std()).astype(int)

    return w


# ===========================
# SAVE ANOMALIES BACK TO MONGO
# ===========================
def push_anomalies(df):
    anomalies = df[df["anomaly"] == 1]
    for _, row in anomalies.iterrows():
        collection.update_one(
            {"timestamp": row["timestamp"]},
            {"$set": {"ml_anomaly": 1}}
        )


# ===========================
# REALTIME LOOP (Streamlit)
# ===========================
placeholder = st.empty()

while True:
    df = load_latest()

    if df.empty:
        placeholder.warning("⏳ Ожидаем данные в Mongo…")
        time.sleep(3)
        continue

    df = detect_anomalies(df)
    push_anomalies(df)

    # hourly aggregation
    df_hour = df.copy()
    df_hour["hour"] = df_hour["timestamp"].dt.floor("H")
    df_hourly = df_hour.groupby("hour").mean().reset_index()

    with placeholder.container():
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        col5 = st.container()

        # CPU
        fig_cpu = go.Figure()
        fig_cpu.add_trace(go.Scatter(
            x=df.timestamp, y=df.cpu, mode="lines", name="CPU"
        ))
        fig_cpu.add_trace(go.Scatter(
            x=df[df.anomaly == 1].timestamp,
            y=df[df.anomaly == 1].cpu,
            mode="markers",
            marker=dict(color="red", size=8),
            name="Anomaly"
        ))
        fig_cpu.update_layout(title="CPU Usage (%)", height=300)
        col1.plotly_chart(fig_cpu, use_container_width=True)

        # MEMORY
        fig_mem = go.Figure()
        fig_mem.add_trace(go.Scatter(
            x=df.timestamp, y=df.memory, mode="lines", name="Memory"
        ))
        fig_mem.add_trace(go.Scatter(
            x=df[df.anomaly == 1].timestamp,
            y=df[df.anomaly == 1].memory,
            mode="markers",
            marker=dict(color="red", size=8),
            name="Anomaly"
        ))
        fig_mem.update_layout(title="Memory Usage (MB)", height=300)
        col2.plotly_chart(fig_mem, use_container_width=True)

        # DISK
        fig_disk = go.Figure()
        fig_disk.add_trace(go.Scatter(
            x=df.timestamp, y=df.disk, mode="lines", name="Disk"
        ))
        fig_disk.add_trace(go.Scatter(
            x=df[df.anomaly == 1].timestamp,
            y=df[df.anomaly == 1].disk,
            mode="markers",
            marker=dict(color="red", size=8),
            name="Anomaly"
        ))
        fig_disk.update_layout(title="Disk I/O (MBps)", height=300)
        col3.plotly_chart(fig_disk, use_container_width=True)

        # NETWORK
        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=df.timestamp, y=df.network, mode="lines", name="Network"
        ))
        fig_net.add_trace(go.Scatter(
            x=df[df.anomaly == 1].timestamp,
            y=df[df.anomaly == 1].network,
            mode="markers",
            marker=dict(color="red", size=8),
            name="Anomaly"
        ))
        fig_net.update_layout(title="Network Usage (MBps)", height=300)
        col4.plotly_chart(fig_net, use_container_width=True)

        # Reconstruction Error Heatmap
        fig_err = px.density_heatmap(
            df,
            x="timestamp",
            y="recon_error",
            color_continuous_scale="Inferno"
        )
        fig_err.update_layout(title="Reconstruction Error Heatmap", height=300)
        col5.plotly_chart(fig_err, use_container_width=True)

    time.sleep(3)
