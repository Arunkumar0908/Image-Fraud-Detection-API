import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.merchant import register_merchant
from app.utils import get_merchants_from_mongo
from sklearn.cluster import DBSCAN
from itertools import combinations
from dotenv import load_dotenv
from app.image_features import compute_similarity_matrix
# from app.objectDetection import classify_with_clip_augmented, handle_featureless
from PIL import Image

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path=env_path)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

app = FastAPI(title="Image Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/register_merchant/")
async def register(
    name: str = Form(...),
    contact: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    front: UploadFile = File(...),
    nameboard: UploadFile = File(...),
    inside: UploadFile = File(...),
):
    front_bytes = await front.read()
    nameboard_bytes = await nameboard.read()
    inside_bytes = await inside.read()

    merchant_info = {"name": name, "contact": contact}

    result = register_merchant(
        front_bytes=front_bytes,
        nameboard_bytes=nameboard_bytes,
        inside_bytes=inside_bytes,
        lat=lat,
        lon=lon,
        merchant_info=merchant_info
    )
    return result

@app.get("/grouped_image_similarity/")
def grouped_image_similarity(radius_km: float = 0.5, similarity_threshold: float = 0.7):
    merchants = get_merchants_from_mongo()
    if not merchants:
        return JSONResponse({"groups": []})

    coords = np.radians([[m['latitude'], m['longitude']] for m in merchants])
    kms_per_radian = 6371.0088
    epsilon = radius_km / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=1, metric='haversine').fit(coords)
    labels = db.labels_

    groups = []
    image_types = ["front", "nameboard", "inside"]

    for cluster_id in set(labels):
        cluster_merchants = [m for m, l in zip(merchants, labels) if l == cluster_id]
        matched_ids = set()
        matches = []

        for m1, m2 in combinations(cluster_merchants, 2):
            best_sim = 0.0
            best_type = None

            for img_type in image_types:
                img1 = m1.get(img_type)
                img2 = m2.get(img_type)

                if not img1 or not img2 or "embeddings" not in img1 or "embeddings" not in img2:
                    continue

                emb1 = np.array(img1["embeddings"])
                emb2 = np.array(img2["embeddings"])

                if emb1.ndim == 1:
                    emb1 = emb1.reshape(1, -1)
                if emb2.ndim == 1:
                    emb2 = emb2.reshape(1, -1)

                sim = compute_similarity_matrix(emb1, emb2)
                if sim > best_sim:
                    best_sim = sim
                    best_type = img_type

            if best_sim >= similarity_threshold:
                matched_ids.update([m1["merchant_id"], m2["merchant_id"]])

            matches.append({
                "primary_merchant_id": m1["merchant_id"],
                "duplicate_merchant_id": m2["merchant_id"],
                "most_similar_image_type": best_type,
                "similarity_percent": round(best_sim * 100, 2),
                "is_match": best_sim >= similarity_threshold
            })

        unique_merchants = [
            {
                "merchant_id": m["merchant_id"],
                "merchant_name": m.get("merchant_name") or m.get("name"),
                "latitude": m["latitude"],
                "longitude": m["longitude"]
            }
            for m in cluster_merchants if m["merchant_id"] not in matched_ids
        ]

        groups.append({
            "area": f"{cluster_merchants[0]['latitude']},{cluster_merchants[0]['longitude']}",
            "matches": matches,
            "unique_merchants": unique_merchants
        })

    return JSONResponse({"groups": groups})

# @app.post("/detect-shop")
# async def detect_shop(file: UploadFile = File(...)):
#     try:
#         with Image.open(file.file) as img:
#             result = handle_featureless(img)
#             if result:
#                 return {"result": result}

#             category = classify_with_clip_augmented(img)
#             return {"result": category}

#     except Exception as e:
#         return {"error": f"Invalid image file: {str(e)}"}
