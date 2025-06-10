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

import json

app = FastAPI()
print("FastAPI app created")

app.state.model_name = "yolo"
app.state.model = download_model_from_gcp('yolo')

print("Model loaded and stored in app.state.model")

@app.get("/")
def root():
    return dict(greeting="Adri le goat")

@app.get("/model")
def model(model_name):
    if model_name != app.state.model_name:
        app.state.model_name = model_name
        app.state.model = download_model_from_gcp(model_name)


#predict grace à un yolo et adapter uniquement à yolo
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model_name = app.state.model_name
    model = app.state.model

    contents = await file.read()

    if model_name == "yolo":
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
    elif model_name == "olympe_model":
        img = Image.open(BytesIO(contents))
        image_path = f"tmp/{file.filename}"
        img.save(image_path)

        # preproc
        image_tensor = resize_with_padding(image_path, (64, 64))
        image_array = (image_tensor.numpy() * 255).astype(np.uint8)

        # MODEL PREDICTION
        model = app.state.model
        prediction = model.predict(image_array)
        return prediction
