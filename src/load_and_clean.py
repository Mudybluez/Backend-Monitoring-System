import pandas as pd

def load_and_clean(path):
    # Загружаем датасет
    df = pd.read_csv(path)

    # Очищаем датасет от лишнего
    columns_needed = [
        "CPU_Usage (%)",
        "Memory_Usage (MB)",
        "Network_Usage (MBps)",
        "Disk_IO (MBps)"
    ]
    df = df[columns_needed]

    # Генерим таймстамп минута 
    df['timestamp'] = pd.date_range(start='2025-01-01 00:00:00', periods=len(df), freq='T')

    return df