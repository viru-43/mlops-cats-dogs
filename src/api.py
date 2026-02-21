import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import time
from pathlib import Path
import logging

from src.model import get_model

logging.basicConfig(level=logging.INFO)

request_count = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# --------------------------------------------------
# Resolve model path safely (works locally + Docker)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pt"

# Load model
model = get_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASS_NAMES = ["Cat", "Dog"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global request_count
    request_count += 1

    start_time = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    latency = time.time() - start_time

    logging.info(
        f"Request #{request_count} | "
        f"Prediction: {CLASS_NAMES[predicted.item()]} | "
        f"Confidence: {confidence.item():.4f} | "
        f"Latency: {latency:.4f}s"
    )

    return {
        "label": CLASS_NAMES[predicted.item()],
        "confidence": float(confidence.item()),
        "latency_seconds": latency
    }


@app.get("/metrics")
def metrics():
    return {
        "total_requests": request_count
    }