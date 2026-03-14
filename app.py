from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load pretrained model
model = MobileNetV2(weights="imagenet")

def process_image(file):
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    
    image = process_image(file.file)

    preds = model.predict(image)

    decoded = decode_predictions(preds, top=1)

    label = decoded[0][0][1]
    confidence = float(decoded[0][0][2])

    return {
        "prediction": label,
        "confidence": confidence
    }