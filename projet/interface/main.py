import pandas as pd
import os
from projet.ml_logic.preprocessor import preprocess_and_save_dataset
from projet.ml_logic.model import train_model, load_and_preprocess_images
from projet.ml_logic.data import download_from_gcp
import numpy as np
from projet.params import LOCAL_DATA_DIR, BUCKET_NAME, TARGET_SIZE


if __name__ == "__main__":
    print("Téléchargement des données prétraitées depuis GCS...")
    # download_from_gcp(prefix_preprocess= "raw_data/ocean waste.v2i.tensorflow", destination_folder=os.path.join(LOCAL_DATA_DIR, "raw/"))
    download_from_gcp(prefix_preprocess= "raw_data/ocean waste.v2i.tensorflow")

    print("Prétraitement des données...")
    for subset in ['train', 'test', 'valid']:
        preprocess_and_save_dataset(
            csv_path=f"data/raw_data/ocean waste.v2i.tensorflow/{subset}/_annotations.csv",
            image_folder=f"data/raw_data/ocean waste.v2i.tensorflow/{subset}/",
            preprocessed_images_root=LOCAL_DATA_DIR,
            target_size=TARGET_SIZE,
            gcp=False,
        )

    size_str = str(TARGET_SIZE[0])
    csv_filename = f"resized_annotations_{size_str}_train.csv"
    csv_path = os.path.join(LOCAL_DATA_DIR, csv_filename)
    df = pd.read_csv(csv_path)
    print(f"{len(df)} annotations chargées")

    df_train = pd.read_csv('/Users/hadrientouchon/code/Hadri2T/wall_e/raw_data/preprocessed_images_64_train/preprocessed_train_annotations_64.csv')

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    df_train['encoded_target'] = encoder.fit_transform(df_train['class'])
    y_train = df_train['encoded_target']

    X_train, y_train = load_and_preprocess_images(df = df_train,
                               image_dir = '/Users/hadrientouchon/code/Hadri2T/wall_e/raw_data/preprocessed_images_64_train',
                               img_size=(64, 64))


    train_model(X_train, y_train, patience = 5, epochs = 2, input_shape=(64, 64, 3))
