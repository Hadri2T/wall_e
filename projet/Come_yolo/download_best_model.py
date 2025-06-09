import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from projet.ml_logic.data import download_from_gcp

from pathlib import Path
from projet.params import *
from google.cloud import storage
from urllib.parse import unquote


if __name__ == '__main__':

    limit=None

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="raw_data/runs_yolo/"))
    if limit:
        blobs = blobs[:limit]
    nb_images = len(blobs)
    print(f"Début du téléchargement de {nb_images} images...")

    for i, blob in enumerate(blobs):
        # Crée un nom de fichier local basé sur l'index ou le nom du blob
        local_filename = unquote(blob.public_url.replace(f'https://storage.googleapis.com/{BUCKET_NAME}/', ''))
        local_filename = os.path.join(local_filename)

        os.makedirs(os.path.dirname(local_filename), exist_ok=True)

        # Télécharge le fichier
        blob.download_to_filename(local_filename)

        print(f"✅ Téléchargé ({i + 1}/{nb_images}) : {local_filename}")
