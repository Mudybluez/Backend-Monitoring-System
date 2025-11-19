import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import json

CSV_PATH = "data/raw/your_dataset.csv"  # заменишь позже

load_dotenv()  # loads .env into os.environ

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")



def load_csv_to_mongo(csv_path):
    print(f"📌 Загружаем CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    # DataFrame → list of dicts
    records = json.loads(df.to_json(orient="records"))

    print("📌 Подключаемся к удалённой MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    print("📌 Чистим старые записи...")
    collection.delete_many({})

    print("📌 Загружаем новые записи...")
    collection.insert_many(records)

    print(f"✅ Загружено {len(records)} записей в удалённый MongoDB!")


if __name__ == "__main__":
    load_csv_to_mongo(CSV_PATH)
