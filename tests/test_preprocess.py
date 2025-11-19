from src.dataset_loader import load_all_datasets
from src.preprocess import normalize_and_merge

dfs = load_all_datasets()

X, df_final = normalize_and_merge(dfs)

print("Preprocess completed!")
print("X shape:", X.shape)
print(df_final.head())
