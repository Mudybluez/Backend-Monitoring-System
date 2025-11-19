import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

def load_mongo_dataset(collection_name="server_metrics"):
    client = MongoClient(MONGO_URI)
    db = client["ml_project"]
    col = db[collection_name]

    data = list(col.find({}, {"_id":0}))
    return pd.DataFrame(data)
