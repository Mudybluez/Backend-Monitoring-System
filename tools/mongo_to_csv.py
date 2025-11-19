import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

OUTPUT_CSV = "data/processed/mongo_export.csv"


def load_mongo():
    print("📌 Подключаемся к облачной MongoDB...")

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print("📌 Загружаем документы...")
    docs = list(collection.find({}, {"_id": 0}))

    df = pd.DataFrame(docs)
    print(f"📌 Форма полученного DataFrame: {df.shape}")

    return df


def convert_to_wide(df):
    print("📌 Обрабатываем данные long-format...")

    # Проверка наличия нужных колонок
    for col in ["timestamp", "metric", "value"]:
        if col not in df.columns:
            print(f"❌ Ошибка: нет колонки '{col}' в MongoDB.")
            return pd.DataFrame()

    # timestamp из ms → datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # pivot long → wide
    df_wide = df.pivot_table(
        index="timestamp",
        columns="metric",
        values="value"
    ).reset_index()

    # anomaly_label
    if "anomaly_label" in df.columns:
        labels = df.groupby("timestamp")["anomaly_label"].max().reset_index()
        df_wide = df_wide.merge(labels, on="timestamp", how="left")

    print("📌 Финальная форма wide-таблицы:", df_wide.shape)
    return df_wide


def export_to_csv(df_wide):
    df_wide.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV успешно сохранён → {OUTPUT_CSV}")


def main():
    df = load_mongo()
    df_wide = convert_to_wide(df)
    export_to_csv(df_wide)


if __name__ == "__main__":
    main()
