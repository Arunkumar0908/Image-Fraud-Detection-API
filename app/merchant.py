from fastapi import APIRouter
from uuid import uuid4
from app.utils import get_nearby_merchants, save_merchant_to_mongo
from app.image_features import extract_image_features, is_duplicate_merchant
from app.merchant_schema import Merchant
from pymongo import MongoClient, GEOSPHERE
import os
from dotenv import load_dotenv

NEARBY_RADIUS_KM = 0.5
router = APIRouter()

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

assert MONGO_URI, "MONGO_URI must be set in .env"
assert DB_NAME, "DB_NAME must be set in .env"
assert MONGO_COLLECTION, "MONGO_COLLECTION must be set in .env"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
merchants_collection = db[MONGO_COLLECTION]
merchants_collection.create_index([("location", GEOSPHERE)])

@router.get("/ping")
def ping():
    return {"message": "pong"}

def flatten_features(features):
    if isinstance(features, dict) and "embeddings" in features:
        emb = features["embeddings"]
        if isinstance(emb[0], list):
            return emb[0]
        return emb
    return features

def register_merchant(front_bytes, nameboard_bytes, inside_bytes, lat, lon, merchant_info):
    front_features = extract_image_features(front_bytes)
    nameboard_features = extract_image_features(nameboard_bytes)
    inside_features = extract_image_features(inside_bytes)

    front_features["embeddings"] = flatten_features(front_features)
    nameboard_features["embeddings"] = flatten_features(nameboard_features)
    inside_features["embeddings"] = flatten_features(inside_features)

    new_imgs = [front_features, nameboard_features, inside_features]

    nearby_merchants = get_nearby_merchants(lat, lon, NEARBY_RADIUS_KM)

    duplicates = is_duplicate_merchant(new_imgs, nearby_merchants, use_ensemble=True)
    if duplicates:
        return {
            "status": "duplicate",
            "details": duplicates,
            "message": "Duplicate merchant detected. Registration blocked."
        }

    merchant_id = str(uuid4())
    data = {
        "merchant_id": merchant_id,
        **merchant_info,
        "latitude": lat,
        "longitude": lon,
        "location": {"type": "Point", "coordinates": [lon, lat]},
        "front": front_features,
        "nameboard": nameboard_features,
        "inside": inside_features
    }

    merchant = Merchant(**data)
    save_merchant_to_mongo(merchant.dict())

    return {
        "status": "success",
        "merchant_id": merchant_id,
        "message": "Merchant successfully registered."
    }
