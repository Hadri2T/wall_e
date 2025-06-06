import pandas as pd
import os
from projet.ml_logic.preprocessor import preprocess_and_save_dataset
from projet.ml_logic.model import model
from projet.ml_logic.data import download_from_gcp
import numpy as np
from projet.params import LOCAL_DATA_DIR, BUCKET_NAME, TARGET_SIZE


if __name__ == "__main__":
    print("Téléchargement des données prétraitées depuis GCS...")
    # download_from_gcp(prefix_preprocess= "raw_data/ocean waste.v2i.tensorflow", destination_folder=os.path.join(LOCAL_DATA_DIR, "raw/"))
    download_from_gcp(prefix_preprocess= "raw_data/ocean waste.v2i.tensorflow", limit = 10)

    print("Prétraitement des données...")
    for subset in ['train', 'test', 'valid']:
        preprocess_and_save_dataset(
            csv_path=f"data/raw_data/ocean waste.v2i.tensorflow/{subset}/annotations.csv",
            image_folder=f"data/raw_data/ocean waste.v2i.tensorflow/{subset}/",
            preprocessed_images_root=LOCAL_DATA_DIR,
            target_size=TARGET_SIZE,
            gcp=False,
            limit = 10
        )

    size_str = str(TARGET_SIZE[0])
    csv_filename = f"resized_annotations_{size_str}_train.csv"
    csv_path = os.path.join(LOCAL_DATA_DIR, csv_filename)
    df = pd.read_csv(csv_path)
    print(df.columns)  # vérifie la liste des colonnes
    print(df.head(1))
    print(f"{len(df)} annotations chargées")
