from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv(r"../.env")
NEWDB_URI = "MONGODB_URI"
client = MongoClient(NEWDB_URI)

try:
    print("databases", client.list_database_names())

except Exception as e:
    print(f"No databases found")

db = client["test"]

db2 = client["doctors"]

# db3 = client ["admin"]

try:
    print("Test database collections :", db.list_collection_names())
    print("\n")
    print("local database collection names are", db2.list_collection_names())

except Exception as es:
    print(f"error in finding database collections", es)

collection = db["doctors"]

doc = collection.find_one()

print(len(doc))
if doc:
    for d in collection.find().limit(10):
        print("-------")
        print("document format:", d,"\n")
        print("-------")

else:
    print("error loading collection format")