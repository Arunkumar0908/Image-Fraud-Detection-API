import os
import math
from pymongo import MongoClient
from functools import lru_cache
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI is not set in environment variables")

@lru_cache()
def get_collection():
    client = MongoClient(MONGO_URI)
    db_name = os.getenv("MONGO_DB_NAME", "merchant_db")
    collection_name = os.getenv("MONGO_COLLECTION", "merchants")
    db = client[db_name]
    return db[collection_name]

def is_nearby(merchant, lat, lon, radius_km):
    if "latitude" not in merchant or "longitude" not in merchant:
        return False

    lat1, lon1 = merchant["latitude"], merchant["longitude"]
    lat2, lon2 = lat, lon
    R = 6371

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance <= radius_km

def get_merchants_from_mongo():
    collection = get_collection()
    return list(collection.find({}, {"_id": 0}))

def save_merchant_to_mongo(data: dict):
    collection = get_collection()
    try:
        if "latitude" in data and "longitude" in data:
            data["location"] = {
                "type": "Point",
                "coordinates": [data["longitude"], data["latitude"]]
            }
        result = collection.insert_one(data)
        print(f"[MongoDB] Inserted merchant ID: {result.inserted_id}")
    except Exception as e:
        print(f"[MongoDB Error] Failed to insert merchant: {e}")
        raise

def get_nearby_merchants(lat, lon, radius_km):
    collection = get_collection()
    radius_m = radius_km * 1000
    try:
        return list(collection.find({
            "location": {
                "$nearSphere": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "$maxDistance": radius_m
                }
            }
        }, {"_id": 0}))
    except Exception as e:
        print(f"[MongoDB Error] Failed to find nearby merchants: {e}")
        return []
