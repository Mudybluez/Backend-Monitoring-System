import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PROCESSED
import os
import joblib
import numpy as np

def normalize_and_merge(dfs):
    merged = []

    for df in dfs:
        df = df.copy()

        # ----------------------------- #
        # 1) TIMESTAMP
        # ----------------------------- #
        ts_cols = [c for c in df.columns if "time" in c.lower()]
        if ts_cols:
            try:
                df["timestamp"] = pd.to_datetime(df[ts_cols[0]])
            except:
                df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="1min")
        else:
            df["timestamp"] = pd.date_range("2023-01-01", periods=len(df), freq="1min")

        # ----------------------------- #
        # 2) AUTO-DETECT METRICS
        # ----------------------------- #
        def pick(cols, *keywords):
            for kw in keywords:
                for col in cols:
                    if kw in col.lower():
                        return col
            return None

        cols = df.columns

        cpu_col    = pick(cols, "cpu")
        mem_col    = pick(cols, "mem", "ram")
        disk_col   = pick(cols, "disk", "storage")
        net_col    = pick(cols, "net", "traffic", "band")

        df_norm = pd.DataFrame()
        df_norm["timestamp"] = df["timestamp"]
        df_norm["cpu"]    = df[cpu_col] if cpu_col else np.nan
        df_norm["memory"] = df[mem_col] if mem_col else np.nan
        df_norm["disk"]   = df[disk_col] if disk_col else np.nan
        df_norm["network"] = df[net_col] if net_col else np.nan

        merged.append(df_norm)

    # ----------------------------- #
    # 3) MERGE ALL DATA
    # ----------------------------- #
    final = pd.concat(merged, ignore_index=True)

    # убираем строки где ВСЕ метрики NaN
    final = final.dropna(subset=["cpu", "memory", "disk", "network"], how="all")

    # сортировка
    final = final.sort_values("timestamp")

    # сохранить raw merged
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    final.to_csv(f"{DATA_PROCESSED}/merged.csv", index=False)

    # ----------------------------- #
    # 4) NORMALIZE
    # ----------------------------- #
    scaler = StandardScaler()

    numeric = final[["cpu", "memory", "disk", "network"]].astype(float)
    numeric = numeric.fillna(numeric.mean())

    scaled = scaler.fit_transform(numeric)

    os.makedirs("models/checkpoints", exist_ok=True)
    joblib.dump(scaler, "models/checkpoints/scaler.pkl")

    return scaled, final
