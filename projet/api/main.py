from fastapi import FastAPI, File, UploadFile
from model.predict import predict_from_image
import shutil
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API de détection de déchets dans l'eau"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Sauvegarder l'image temporairement
    image_path = f"tmp_{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Appeler le modèle
    prediction = predict_from_image(image_path)

    # Supprimer l'image après
    os.remove(image_path)

    return {"predictions": prediction}
