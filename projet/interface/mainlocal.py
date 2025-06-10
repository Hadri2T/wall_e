# main.py – version locale sans GCP obligatoire
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from projet.ml_logic.preprocessor import preprocess_and_save_dataset
from projet.ml_logic.data import download_from_gcp
from projet.ml_logic.oneclass import load_and_preprocess_images as load_oneclass_data, train_oneclassmodel
from projet.ml_logic.multiclass import load_and_preprocess_images as load_multiclass_data, train_multiclassmodel
from projet.params import LOCAL_DATA_DIR, TARGET_SIZE
from sklearn.preprocessing import LabelEncoder
from keras.models import save_model

USE_GCP = False  # ✉️ Active False pour travailler localement

if __name__ == "__main__":

    # 1. Vérification ou prétraitement des données
    if not os.path.isdir("data/raw_data/ocean waste.v2i.tensorflow/train"):
        if USE_GCP:
            print("📄 Téléchargement des données depuis GCS...")
            for split in ["train", "valid", "test"]:
                download_from_gcp(prefix_preprocess=f"raw_data/ocean waste.v2i.tensorflow/{split}/")
        else:
            raise FileNotFoundError("Les données ne sont pas présentes en local et USE_GCP=False")
    else:
        print("📁 Données déjà présentes localement.")

    if not os.path.isdir("data/preprocessed_images_64_train"):
        print("📃 Prétraitement des données...")
        for subset in ['train', 'test', 'valid']:
            preprocess_and_save_dataset(
                csv_path=f"data/raw_data/ocean waste.v2i.tensorflow/{subset}/_annotations.csv",
                image_folder=f"data/raw_data/ocean waste.v2i.tensorflow/{subset}/",
                preprocessed_images_root=LOCAL_DATA_DIR,
                target_size=TARGET_SIZE,
                gcp=False,
            )
    else:
        print("🔄 Données déjà prétraitées.")

    # 2. Chargement du CSV prétraité
    csv_path = os.path.join(LOCAL_DATA_DIR, f"preprocessed_images_{TARGET_SIZE[0]}_train", f"resized_annotations_{TARGET_SIZE[0]}_train.csv")
    df_train = pd.read_csv(csv_path)
    print(f"🔢 {len(df_train)} lignes chargées depuis {csv_path}")

    # 3. Encodage des labels
    encoder = LabelEncoder()
    df_train['encoded_target'] = encoder.fit_transform(df_train['class'])
    y_train = df_train['encoded_target']

    # 4. Choix du modèle
    choice = input("Quel modèle veux-tu entraîner ? (oneclass / multiclass) : ").strip().lower()
    if choice not in ["oneclass", "multiclass"]:
        print("❌ Choix invalide. Veuillez entrer 'oneclass' ou 'multiclass'.")
        exit()

    # 5. Chargement des images
    image_dir = f"data/preprocessed_images_64_train"
    if choice == "oneclass":
        X_train, y_train = load_oneclass_data(df_train, image_dir=image_dir, img_size=TARGET_SIZE)
        model = train_oneclassmodel(X_train, y_train, patience=5, epochs=5, input_shape=(*TARGET_SIZE, 3))
        model_save_path = os.path.join(LOCAL_DATA_DIR, "one_class_model.h5")
    else:
        X_train, y_train = load_multiclass_data(df_train, image_dir=image_dir, img_size=TARGET_SIZE)
        model = train_multiclassmodel(X_train, y_train, patience=5, epochs=5, input_shape=(*TARGET_SIZE, 3))
        model_save_path = os.path.join(LOCAL_DATA_DIR, "multiclass_model.h5")

    # 6. Sauvegarde
    save_model(model, model_save_path)
    print(f"✅ Modèle sauvegardé à : {model_save_path}")
