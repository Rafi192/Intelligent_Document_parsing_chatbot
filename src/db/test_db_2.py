# test_connection.py
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
DATABASE_NAME = os.getenv("MONGODB_DATABASE", "test")


print(f"Database: {DATABASE_NAME}")

client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

collections = db.list_collection_names()
print(f"\nCollections in '{DATABASE_NAME}':")
for coll in collections:
    count = db[coll].count_documents({})
    print(f"  - {coll}: {count} documents")

client.close()