import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from projet.ml_logic.preprocessor import preprocess_and_save_dataset
from projet.ml_logic.oneclass import train_oneclassmodel, load_and_preprocess_images
from projet.ml_logic.data import download_from_gcp
import numpy as np
from projet.params import LOCAL_DATA_DIR, BUCKET_NAME, TARGET_SIZE
from sklearn.preprocessing import LabelEncoder
from keras.models import save_model


if __name__ == "__main__":

    # download_from_gcp(prefix_preprocess= "raw_data/ocean waste.v2i.tensorflow", destination_folder=os.path.join(LOCAL_DATA_DIR, "raw/"))
    if not os.path.isdir('data/raw_data/ocean waste.v2i.tensorflow/train'):
        print("Téléchargement des données depuis GCS...")
        for split in ["train", "valid", "test"]:
            download_from_gcp(prefix_preprocess=f"raw_data/ocean waste.v2i.tensorflow/{split}/")
    else:
        print("📁 Données déjà présentes localement.")

    if os.path.isdir("data/preprocessed_images_64_train"):
        print("Data already exists")
    else:
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
    csv_filename = f"preprocessed_images_{size_str}_train/resized_annotations_{size_str}_train.csv"
    csv_path = os.path.join(LOCAL_DATA_DIR, csv_filename)
    print(csv_path)
    df = pd.read_csv(csv_path)
    print(f"{len(df)} annotations chargées")

    print(f"{csv_path = }")

    df_train = pd.read_csv('data/preprocessed_images_64_train/resized_annotations_64_train.csv')

    print(df_train)

    encoder = LabelEncoder()
    df_train['encoded_target'] = encoder.fit_transform(df_train['class'])
    y_train = df_train['encoded_target']

    X_train, y_train = load_and_preprocess_images(df = df_train,
                               image_dir = 'data/preprocessed_images_64_test',
                               img_size=(64, 64))


    model_one_class = train_oneclassmodel(X_train, y_train, patience = 5, epochs = 1, input_shape=(64, 64, 3))

    model_one_class_save_path = os.path.join(LOCAL_DATA_DIR, "one_class_model.h5")
    save_model(model_one_class, model_one_class_save_path)
    print(f"✅ Modèle sauvegardé à : {model_one_class_save_path}")
