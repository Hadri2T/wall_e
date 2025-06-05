from ml_logic import resize_with_padding, resize_bounding_boxes, filter_images
import pandas as pd
import os
from ml_logic.preprocessor import preprocess_and_save_dataset
from ml_logic.model import model
from ml_logic.data import download_from_gcp
import pandas as pd
from params import GCP_PROJECT, LOCAL_DATA_DIR, BUCKET_NAME, TARGET_SIZE


if __name__ == "__main__":
    print("Téléchargement des données prétraitées depuis GCS...")
    download_from_gcp(prefix_preprocess=GCP_PROJECT, destination_folder=LOCAL_DATA_DIR)

    size_str = str(TARGET_SIZE[0])  # ou "_".join(map(str, TARGET_SIZE)) si tu veux les deux dimensions
    csv_filename = f"resized_annotations_{size_str}_train.csv"
    csv_path = os.path.join(LOCAL_DATA_DIR, csv_filename)
    df = pd.read_csv(csv_path)
    print(f"{len(df)} annotations chargées")
