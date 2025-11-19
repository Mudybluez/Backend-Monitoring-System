import numpy as np

def add_anomalies(df):
    """
    Добавляет синтетические spikes/drops для CPU, Memory, Network и Disk
    """
    df['anomaly_label'] = 0  # 0 = нормальное, 1 = аномалия

    # CPU spikes (1% записей)
    cpu_idx = df.sample(frac=0.01).index
    df.loc[cpu_idx, "CPU_Usage (%)"] *= np.random.uniform(1.5, 3)
    df.loc[cpu_idx, "anomaly_label"] = 1

    # Memory drops (1%)
    mem_idx = df.sample(frac=0.01).index
    df.loc[mem_idx, "Memory_Usage (MB)"] *= np.random.uniform(0.2, 0.5)
    df.loc[mem_idx, "anomaly_label"] = 1

    # Disk spikes (0.5%)
    disk_idx = df.sample(frac=0.005).index
    df.loc[disk_idx, "Disk_IO (MBps)"] *= np.random.uniform(1.5, 2.5)
    df.loc[disk_idx, "anomaly_label"] = 1

    # Network spikes/drops (0.5%)
    net_idx = df.sample(frac=0.005).index
    df.loc[net_idx, "Network_Usage (MBps)"] *= np.random.uniform(1.5, 2.0)
    df.loc[net_idx, "anomaly_label"] = 1

    return df
