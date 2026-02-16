import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import time

from src.model import get_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()

# Load model
model = get_model()
model.load_state_dict(torch.load("models/model.pt", map_location=DEVICE))
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
    start_time = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    latency = time.time() - start_time

    return {
        "label": CLASS_NAMES[predicted.item()],
        "confidence": float(confidence.item()),
        "latency_seconds": latency
    }
