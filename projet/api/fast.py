import pandas as pd
import numpy as np
import cv2

from projet.ml_logic.data import download_model_from_gcp
from projet.ml_logic.preprocessor import resize_with_padding
from fastapi import UploadFile, File
from PIL import Image
from io import BytesIO

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
print("FastAPI app created")

app.state.model = download_model_from_gcp('yolo')

print("Model loaded and stored in app.state.model")

#predict grace à un yolo et adapter uniquement à yolo
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image uploadée
    contents = await file.read()

    # Convertir en image RGB
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Convertir en tableau numpy pour YOLO
    img_np = np.array(image)
    processed_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Prédiction avec YOLO
    model = app.state.model
    results = model.predict(processed_img)

    boxes = results[0].boxes
    waste_categories = boxes.cls.numpy().tolist()
    confidence_score = boxes.conf.numpy().tolist()
    bounding_boxes = boxes.xyxy.numpy().tolist()

    last_prediction = {
        "waste_categories": waste_categories,
        "confidence_score": confidence_score,
        "bounding_boxes": bounding_boxes
    }
    return last_prediction

@app.get("/")
def root():
    return dict(greeting="Adri le goat")
