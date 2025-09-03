# 🛡️ Image Fraud Detection API

A FastAPI-based backend service to detect duplicate or fraudulent merchant images using **CLIP embeddings**, **image hashing**, and **geo-based clustering**.

## Features
- Merchant registration with multiple images
- Image embeddings via CLIP & ResNet50 for similarity comparison
- Perceptual, average, difference, and wavelet hashing for robustness
- DBSCAN clustering for geo-location grouping
- Duplicate detection with configurable similarity thresholds
- Shop category classification with CLIP + augmentations
- REST API built with **FastAPI** + **MongoDB**

## Tech Stack
- **Python 3.9+**
- **FastAPI**
- **MongoDB**
- **scikit-learn**
- **PyTorch (ResNet-50)**
- **NumPy / PIL**
- **CLIP (OpenAI)**

## Setup

###1. Clone repo
```bash
git clone https://github.com/Arunkumar0908/Image-Fraud-Detection-API.git
cd Image-Fraud-Detection-API

###2. Create virtual env

python3 -m venv venv
source venv/bin/activate

###3. Install dependencies

python3 -m pip install --upgrade pip
pip install torch==2.2.0 torchvision==0.17.0
pip install -r requirements.txt  
pip install git+https://github.com/openai/CLIP.git

###4. Configure environment

Create a .env file in root:

MONGO_URI=mongodb://localhost:27017
DB_NAME=image_fraud
MONGO_COLLECTION=merchants

###5. Run server

uvicorn app.main:app --host 0.0.0.0 --port 8001 --env-file .env
API Endpoints

POST /register_merchant/ → Register merchant with images

GET /grouped_image_similarity/ → Group merchants by geo + similarity

POST /detect-shop → Classify shop type
