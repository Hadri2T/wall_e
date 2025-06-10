
from ultralytics import YOLO
from google.cloud import storage
import os
from urllib.parse import unquote

# Remplacez ces variables par les vôtres si nécessaire
LOCAL_DATA_DIR = "data"
BUCKET_NAME = "wall-e-bucket1976"

def download_model_from_gcp_yolo():
    """
    Télécharge le fichier best.pt depuis GCP et le charge avec Ultralytics YOLO.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Spécifie le chemin exact du modèle dans le bucket
    blob_path = "raw_data/runs_yolo/train15/weights/best.pt"
    blob = bucket.blob(blob_path)

    # Détermine le chemin local pour stocker le modèle
    local_model_path = os.path.join(LOCAL_DATA_DIR, "best.pt")
    os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

    # Télécharge le modèle
    blob.download_to_filename(local_model_path)
    print(f"✅ Modèle YOLO téléchargé depuis GCP : {blob_path}")

    # Charge le modèle YOLO
    model = YOLO(local_model_path)
    print("✅ Modèle YOLO chargé avec succès.")
    print 
    return model

if __name__ == "__main__":
    model = download_model_from_gcp_yolo()
    # Test : à remplacer par le chemin de ton image locale
    # results = model('path/to/your/image.jpg', show=True)
