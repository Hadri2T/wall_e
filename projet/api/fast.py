import numpy as np
import cv2

from projet.ml_logic.data import download_model_from_gcp
from fastapi import UploadFile, File, FastAPI
from PIL import Image, ImageOps
from io import BytesIO

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load both models ===
app.state.models = {
    "yolo": download_model_from_gcp('yolo'),
    "olympe_model": download_model_from_gcp("olympe_model")

}

# Set default model
app.state.current_model = "olympe_model"
print("Models loaded and default set to Olympe")

@app.get("/")
def root():
    return {"message": "API de détection de déchets : YOLO et CNN supportés."}

@app.get("/model")
def choose_model(model_name: str):
    if model_name not in app.state.models:
        return {"error": f"Model '{model_name}' not found."}
    app.state.current_model = model_name
    return {"message": f"Model '{model_name}' selected."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    img_np = np.array(image)

    model_name = app.state.current_model
    model = app.state.models[model_name]

    if model_name == "yolo":
        processed_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = model.predict(processed_img)
        boxes = results[0].boxes
        return {
            "waste_categories": boxes.cls.numpy().tolist(),
            "confidence_score": boxes.conf.numpy().tolist(),
            "bounding_boxes": boxes.xyxy.numpy().tolist()
        }

    elif model_name == "olympe_model":
        # img_resized = image.resize((64, 64))


        # Taille actuelle
        orig_width, orig_height = image.size

        # Ratio de redimensionnement
        ratio = min(64 / orig_width, 64 / orig_height)
        new_size = (int(orig_width * ratio), int(orig_height * ratio))

        # Redimensionne l’image sans déformation
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Ajout du padding pour correspondre à la taille cible
        padded_image = ImageOps.pad(
            resized_image,
            (64, 64),
            method=Image.Resampling.LANCZOS,
            color=(0, 0, 0),
            centering=(0.5, 0.5)  # centre l’image dans le canvas
        )

        # img_array = np.array(img_resized) / 255.0

        img_array = np.array(padded_image) / 255.0

        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return {"prediction": prediction[0].tolist()}
