from pymongo import MongoClient

def upload_to_mongo(df, mongo_uri, db_name="cloud_metrics_db", collection_name="metrics"):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    collection.insert_many(df.to_dict(orient="records"))
    print(f"Данные успешно загружены в MongoDB: {db_name}.{collection_name}")