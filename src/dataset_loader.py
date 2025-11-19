import os
import pandas as pd
from src.config import DATA_RAW

def load_all_datasets():
    dfs = []

    for root, dirs, files in os.walk(DATA_RAW):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                print("Loading:", path)
                
                try:
                    df = pd.read_csv(path)
                    dfs.append(df)
                except:
                    df = pd.read_csv(path, encoding="latin1", engine="python")
                    dfs.append(df)

    return dfs
