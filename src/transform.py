import pandas as pd

def to_long_format(df, provider="programmer3_cloud_dataset"):
    
    records = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        for metric in ["CPU_Usage (%)", "Memory_Usage (MB)", "Disk_IO (MBps)", "Network_Usage (MBps)"]:
            records.append({
                "timestamp": ts,
                "metric": metric,
                "value": float(row[metric]),
                "provider": provider,
                "anomaly_label": int(row['anomaly_label'])
            })
    return pd.DataFrame(records)
