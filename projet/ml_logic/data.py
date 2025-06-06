import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from pathlib import Path
from projet.params import *
from google.cloud import storage
from urllib.parse import unquote

# Upload local → GCP
def upload_to_gcp(from_folder):
    client = storage.Client()
    bucket = client.bucket("wall-e-bucket1976")

    # Parcourt tous les fichiers présents dans le dossier local
    for file in Path(from_folder).rglob('*'):
        if file.is_file():
            destination_path = str(file.relative_to("."))
            print(destination_path)
            blob = bucket.blob(destination_path)
            blob.upload_from_string(file.read_bytes())
            print(f"✅ Uploadé : {destination_path} → gs://{BUCKET_NAME}/{destination_path}")

# Download GCP → local
# def download_from_gcp(prefix_preprocess, destination_folder):
def download_from_gcp(prefix_preprocess, limit = None):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix_preprocess))
    if limit:
        blobs = blobs[:limit]
    nb_images = len(blobs)
    print(f"Début du téléchargement de {nb_images} images...")

    for i, blob in enumerate(blobs):
        # Crée un nom de fichier local basé sur l'index ou le nom du blob
        local_filename = unquote(blob.public_url.replace(f'https://storage.googleapis.com/{BUCKET_NAME}/', ''))
        local_filename = os.path.join(LOCAL_DATA_DIR, local_filename)

        os.makedirs(os.path.dirname(local_filename), exist_ok=True)

        # Télécharge le fichier
        blob.download_to_filename(local_filename)

        print(f"✅ Téléchargé ({i + 1}/{nb_images}) : {local_filename}")




if __name__ == "__main__":
    folder = input("Folder name to upload to GCP : ")
    upload_to_gcp(folder)
