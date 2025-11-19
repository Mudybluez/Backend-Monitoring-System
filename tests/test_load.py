# tests/test_load.py

from src.dataset_loader import load_all_datasets

dfs = load_all_datasets()
print(f"Найдено датасетов: {len(dfs)}")

for i, df in enumerate(dfs):
    print(f"\n--- Датасет #{i+1} ---")
    print(df.head())
