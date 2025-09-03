import io
import torch
import numpy as np
import imagehash
from PIL import Image, ExifTags, ImageEnhance, ImageFilter, ImageOps
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from typing import List, Dict
import logging
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
resnet.eval()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@lru_cache(maxsize=128)
def get_orientation_key():
    return next((key for key, val in ExifTags.TAGS.items() if val == 'Orientation'), None)

def correct_orientation(image: Image.Image) -> Image.Image:
    try:
        exif = getattr(image, "_getexif", lambda: None)()
        if exif is not None:
            orientation_key = get_orientation_key()
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                rotation_map = {3: 180, 6: 270, 8: 90}
                if orientation in rotation_map:
                    image = image.rotate(rotation_map[orientation], expand=True)
    except Exception as e:
        logger.warning(f"Error correcting orientation: {e}")
    return image

def extract_embeddings_batch(images: List[Image.Image], model, preprocess, device="cuda") -> np.ndarray:
    batch_tensors = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        tensor = preprocess(img)
        batch_tensors.append(tensor)
    batch = torch.stack(batch_tensors).to(device)
    
    with torch.no_grad():
        features = model(batch)
        features = features.view(features.size(0), -1)
        embeddings = features.cpu().numpy()
    
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    embeddings = embeddings / norms
    return embeddings

def smart_augment_image(image: Image.Image, max_augmentations: int = 24) -> List[Image.Image]:
    augmented_images = [image]
    rotation_angles = list(range(5, 360, 5))
    rotation_augmentations = [lambda img, angle=angle: img.rotate(angle, expand=True) for angle in rotation_angles]

    other_augmentations = [
        lambda img: img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        lambda img: img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        lambda img: ImageEnhance.Brightness(img).enhance(0.7),
        lambda img: ImageEnhance.Brightness(img).enhance(1.3),
        lambda img: ImageEnhance.Contrast(img).enhance(0.8),
        lambda img: ImageEnhance.Contrast(img).enhance(1.2),
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4)),
        lambda img: ImageEnhance.Sharpness(img).enhance(2.0),
        lambda img: img.transform(img.size, Image.Transform.AFFINE, (1, 0.3, -30, 0.1, 1, -20)),
        lambda img: img.transform(img.size, Image.Transform.AFFINE, (1, -0.2, 20, 0.3, 1, -10)),
        lambda img: img.crop((10, 10, img.width - 10, img.height - 10)).resize(img.size),
        lambda img: img.crop((20, 0, img.width, img.height - 20)).resize(img.size),
        lambda img: img.filter(ImageFilter.GaussianBlur(radius=1)),
        lambda img: ImageOps.invert(img.convert("RGB")),
    ]
    all_augmentations = rotation_augmentations + other_augmentations
    selected_augmentations = random.sample(all_augmentations, min(max_augmentations - 1, len(all_augmentations)))
    for aug in selected_augmentations:
        try:
            augmented_images.append(aug(image.copy()))
        except Exception as e:
            logger.warning(f"[Warning] Augmentation failed: {e}")
    return augmented_images

def extract_embeddings_from_augmented(image: Image.Image, model, preprocess, device=None) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    image = correct_orientation(image)
    augmented_images = smart_augment_image(image)
    embeddings = extract_embeddings_batch(augmented_images, model, preprocess, device)
    return embeddings

def extract_multiple_hashes(image: Image.Image) -> Dict[str, str]:
    return {
        "average_hash": str(imagehash.average_hash(image)),
        "perceptual_hash": str(imagehash.phash(image)),
        "difference_hash": str(imagehash.dhash(image)),
        "wavelet_hash": str(imagehash.whash(image))
    }

def extract_image_features(image_bytes: bytes, model=None, preprocess=None, device=None) -> Dict:
    if model is None:
        model = resnet
    if preprocess is None:
        preprocess = globals()['preprocess']
    if device is None:
        device = globals()['device']
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = correct_orientation(image)
    embeddings = extract_embeddings_from_augmented(image, model, preprocess, device)
    hashes = extract_multiple_hashes(image)
    return {
        "embeddings": embeddings.tolist(),
        "hashes": hashes,
        "embedding_mean": np.mean(embeddings, axis=0).tolist(),
        "embedding_std": np.std(embeddings, axis=0).tolist()
    }

def compute_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    if len(embeddings1) == 0 or len(embeddings2) == 0:
        return 0.0

    emb1 = np.atleast_2d(embeddings1)
    emb2 = np.atleast_2d(embeddings2)

    similarity_matrix = sklearn_cosine_similarity(emb1, emb2)
    return float(np.max(similarity_matrix))

def compute_hash_similarity(hashes1: Dict[str, str], hashes2: Dict[str, str]) -> Dict[str, float]:
    similarities = {}
    for hash_type in hashes1.keys():
        if hash_type in hashes2:
            try:
                hash1 = imagehash.hex_to_hash(hashes1[hash_type])
                hash2 = imagehash.hex_to_hash(hashes2[hash_type])
                distance = hash1 - hash2
                similarities[hash_type] = max(0, 1 - distance / 64.0)
            except Exception as e:
                logger.warning(f"Error computing {hash_type} similarity: {e}")
                similarities[hash_type] = 0.0
        else:
            similarities[hash_type] = 0.0
    return similarities

def is_duplicate_merchant(
    new_imgs: List[Dict], 
    existing_merchants: List[Dict], 
    phash_threshold: float = 0.70,
    embedding_threshold: float = 0.75,
    use_ensemble: bool = True
) -> List[Dict]:
    duplicate_matches = []
    image_types = ["front", "nameboard", "inside"]

    for merchant in existing_merchants:
        for idx, img_type in enumerate(image_types):
            if idx >= len(new_imgs):
                continue
            new_img = new_imgs[idx]
            db_img = merchant.get(img_type)
            if not db_img or not isinstance(db_img, dict):
                continue

            db_embeddings = np.array(db_img.get("embeddings", []))
            db_hashes = db_img.get("hashes", {})
            new_embeddings = np.array(new_img.get("embeddings", []))
            new_hashes = new_img.get("hashes", {})

            if len(db_embeddings) == 0 or len(new_embeddings) == 0:
                continue

            embedding_sim = compute_similarity_matrix(new_embeddings, db_embeddings)
            hash_similarities = compute_hash_similarity(new_hashes, db_hashes)

            if use_ensemble:
                weights = {
                    "embedding": 0.8,
                    "perceptual_hash": 0.1,#structure 
                    "average_hash": 0.05,#brightness
                    "difference_hash": 0.025,#edges #shapes
                    "wavelet_hash": 0.025#texture #patterns
                }
                ensemble_score = (
                    weights["embedding"] * embedding_sim +
                    weights["perceptual_hash"] * hash_similarities.get("perceptual_hash", 0) +
                    weights["average_hash"] * hash_similarities.get("average_hash", 0) +
                    weights["difference_hash"] * hash_similarities.get("difference_hash", 0) +
                    weights["wavelet_hash"] * hash_similarities.get("wavelet_hash", 0)
                )
                is_duplicate = ensemble_score >= embedding_threshold
                final_similarity = ensemble_score
            else:
                best_hash_sim = max(hash_similarities.values()) if hash_similarities else 0
                is_duplicate = best_hash_sim >= phash_threshold or embedding_sim >= embedding_threshold
                final_similarity = max(embedding_sim, best_hash_sim)

            risk_percent = round(final_similarity * 100, 2)

            if is_duplicate:
                duplicate_matches.append({
                    "existing_merchant_id": merchant.get("merchant_id"),
                    "existing_merchant_name": merchant.get("merchant_name") or merchant.get("name"),
                    "existing_latitude": merchant.get("latitude"),
                    "existing_longitude": merchant.get("longitude"),
                    "matched_image_type": img_type,
                    "embedding_similarity": float(embedding_sim),
                    "hash_similarities": hash_similarities,
                    "final_similarity": float(final_similarity),
                    "risk_percent": risk_percent,
                    "message": f"Duplicate detected for {img_type} image with {risk_percent}% similarity."
                })
    return duplicate_matches
