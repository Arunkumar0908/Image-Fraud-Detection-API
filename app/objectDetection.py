from PIL import Image
import torch
from torch import Tensor
from typing import cast
import numpy as np
from app.image_features import correct_orientation, smart_augment_image, preprocess
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

shop_categories = [
    "photo of a aquarium with fish and plants",
    "photo of a bakery with pastries on display",
    "photo of a bar with bottles and stools",
    "photo of a barber shop with chairs and mirrors",
    "photo of a beauty salon with styling chairs",
    "photo of a bookstore with books on shelves",
    "photo of a car dealership with cars on display",
    "photo of a car wash with vehicles being cleaned",
    "photo of a clothing store with mannequins",
    "photo of a coffee shop with tables and counter",
    "photo of a dairy shop with milk and cheese",
    "photo of a department store with aisles",
    "photo of a electronics repair shop with tools and devices",
    "photo of a electronics store with gadgets on shelves",
    "photo of a fast food restaurant with counters",
    "photo of a fish market with seafood",
    "photo of a florist with flower arrangements",
    "photo of a flower shop with floral arrangements",
    "photo of a furniture store with sofas and chairs",
    "photo of a gas station with pumps",
    "photo of a gift shop with souvenirs",
    "photo of a goods store with various products",
    "photo of a grocery shop with fruits and vegetables",
    "photo of a gym with fitness equipment",
    "photo of a hardware store with tools",
    "photo of a hospital with medical equipment",
    "photo of a internet cafe with computers",
    "photo of a jewelry store with glass display cases",
    "photo of a juice shop with fresh fruits",
    "photo of a laboratory with scientific equipment",
    "photo of a liquor store with shelves of alcohol",
    "photo of a meat shop with various cuts of meat",
    "photo of a mechanic shop with tools and vehicles",
    "photo of a mobile store with smartphones and accessories",
    "photo of a movie theater with seats and screen",
    "photo of a music store with instruments",
    "photo of a office supply store with desks and supplies",
    "photo of a pawnbroker with items on display",
    "photo of a petrol station with fuel pumps",
    "photo of a pharmacy with medicine racks",
    "photo of a restaurant with dining area",
    "photo of a roadside shop with various goods",
    "photo of a sculpture with art supplies",
    "photo of a shoe store with shoes on display",
    "photo of a spa with massage tables",
    "photo of a sports store with sports equipment",
    "photo of a stationery shop with paper and supplies",
    "photo of a storage unit with boxes",
    "photo of a studio with cameras and lights",
    "photo of a supermarket with grocery aisles",
    "photo of a tattoo parlor with tattoo chairs and designs",
    "photo of a tea shop with teapots and cups",
    "photo of a toy store with toys and games",
    "photo of a veterinary clinic with animals and equipment",
    "photo of a video game store with games and consoles",
    "photo of a woodworking shop with tools and wood",
    "photo of an art gallery with paintings and sculptures",
    "photo of an cyber cafe with computer systems",
    "photo of an ice cream shop with cones and scoops",
    "photo of an optical store with glasses and frames",
    "photo of an unknown shop"
]

with torch.no_grad():
    text_inputs = clip.tokenize(shop_categories).to(device)
    text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def is_featureless(img: Image.Image, threshold: float = 10.0) -> bool:
    arr = np.array(img.convert("L"))
    return arr.var() < threshold

def handle_featureless(img: Image.Image):
    if is_featureless(img):
        return "photo of an unknown shop"
    return None

def classify_with_clip_augmented(img: Image.Image) -> str:
    img = correct_orientation(img)
    crops = smart_augment_image(img, max_augmentations=12)
    embeddings_list = []
    with torch.no_grad():
        for crop in crops:
            tensor: Tensor = cast(Tensor, preprocess(crop.convert("RGB")))
            tensor = tensor.unsqueeze(0).to(device)
            emb = clip_model.encode_image(tensor)
            emb /= emb.norm(dim=-1, keepdim=True)
            embeddings_list.append(emb)
    
    image_features = torch.mean(torch.stack(embeddings_list), dim=0)

    similarities = (image_features @ text_features.T).squeeze(0)
    best_idx = similarities.argmax().item()

    if similarities[best_idx].item() < 0.15: 
        return "photo of an unknown shop"
    return shop_categories[best_idx]