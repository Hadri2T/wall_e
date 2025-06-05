<<<<<<< HEAD
def load_data(filepath="raw_data/valid_annotations.csv", max_objects=1):
    """
    Load and preprocess image annotation data from a CSV file.

    This function reads image annotation data, filters images based on the maximum number of objects allowed,
    and creates a binary target column indicating whether the object class is 'Plastic'.

    Args:
        filepath (str): Path to the CSV file containing annotation data. Defaults to "raw_data/valid_annotations.csv".
        max_objects (int): Maximum number of objects allowed per image. Defaults to 1.

    Returns:
        pandas.DataFrame: A DataFrame containing filtered image annotations with an additional 'target' column,
        where 'target' is True if the class is 'Plastic', otherwise False.
    """
    df = filter_images(filepath, max_objects=max_objects)
    df['target'] = (df['class'] == 'Plastic')
    return df
=======
from pathlib import Path
import os
from projet.params import *
from google.cloud import storage

# Upload local → GCP
def upload_to_gcp(from_folder):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Parcourt tous les fichiers présents dans le dossier local
    for file in Path(from_folder).rglob('*'):
        if file.is_file():
            destination_path = str(file.relative_to("."))
            blob = bucket.blob(destination_path)
            blob.upload_from_string(file.read_bytes())
            print(f"✅ Uploadé : {destination_path} → gs://{BUCKET_NAME}/{destination_path}")

# Download GCP → local
def download_from_gcp(prefix_preprocess, destination_folder):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=prefix_preprocess))
    nb_images = len(blobs)
    print("Début du téléchargement de {nb_images} images...")

    for i, blob in enumerate(blobs):
        # Crée un nom de fichier local basé sur l'index ou le nom du blob
        local_filename = os.path.join(destination_folder, f"{blob.name}")

        # Télécharge le fichier
        if not os.path.exists(os.path.dirname(local_filename)):
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
        blob.download_to_filename(local_filename)
        print(f"Téléchargé : image ({i + 1}/{nb_images})")

if __name__ == "__main__":
    # download_from_gcp("processed", "processed_data/")
    upload_to_gcp("test_blobs")
>>>>>>> 4ab36519bd2713f1f54d19188dce1db5e5b83b21
