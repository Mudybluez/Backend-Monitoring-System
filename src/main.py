from src.load_and_clean import load_and_clean
from src.enrich_anomalies import add_anomalies
from src.transform import to_long_format
from src.mongo_upload import upload_to_mongo


df = load_and_clean(r"data\cloud_resource_allocation_dataset.csv")
df = add_anomalies(df)
long_df = to_long_format(df)

# MongoDB URI
mongo_uri = "mongodb+srv://alex_madi:alex424aga@cluster0.bc8xa9d.mongodb.net/?appName=Cluster0"
upload_to_mongo(long_df, mongo_uri)
# Имя базы и коллекции

