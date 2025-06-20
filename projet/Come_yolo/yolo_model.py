#code compelt avec une fonction qui demande si je veux changer le nombre d'epoch et la resolution d'image :
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from ultralytics import YOLO
from roboflow import Roboflow
from projet.ml_logic.data import upload_to_gcp



def download_dataset_roboflow():
    rf = Roboflow(api_key="aRxK3L3jdKNj24RDgvNr")
    project = rf.workspace("research-7dj8h").project("ocean-waste")
    dataset = project.version(1).download("yolov8")
    return dataset

def get_training_params(default_epochs=3, default_imgsz=64):
    response = input("Souhaitez-vous modifier les paramètres d'entraînement ? (y/n) : ").strip().lower()

    if response == 'y':
        try:
            epochs = int(input(f"Nombre d'epochs (défaut = {default_epochs}) : ").strip())
        except ValueError:
            print("Entrée invalide, utilisation de la valeur par défaut.")
            epochs = default_epochs

        try:
            imgsz = int(input(f"Taille des images (imgsz) (défaut = {default_imgsz}) : ").strip())
        except ValueError:
            print("Entrée invalide, utilisation de la valeur par défaut.")
            imgsz = default_imgsz
    else:
        epochs = default_epochs
        imgsz = default_imgsz

    return epochs, imgsz

def train_yolo(data_yaml_path, epochs=3, imgsz=64):
    model = YOLO("yolov8n.pt")  # Démarre avec le modèle pré-entraîné
    model.train(
        data='/Users/olympedupont/code/Hadri2T/wall_e/projet/ml_logic/data_yolo/data.yaml',
        epochs=epochs,
        imgsz=imgsz,
        save_period=5,
        project='raw_data/runs_yolo',
        batch=16
    )

    # Sauvegarder le modèle entraîné
    model_path = "raw_data/runs_yolo/train/weights/best.pt"
    print(f"Modèle entraîné sauvegardé ici : {model_path}")


if __name__ == '__main__':
    data_yaml_path = '/Users/comelubrano/code/Hadri2T/wall_e/raw_data/ocean-waste-1/ocean waste.v2i.yolov11/data.yaml'

    while True:
        user_input = input('Tu veux télécharger le dataset ? (y/n) : ').strip().lower()
        if user_input == 'y':
            dataset = download_dataset_roboflow()
            data_yaml_path = dataset.location + "/data.yaml"
            break
        elif user_input == 'n':
            print("Le dataset ne sera pas téléchargé.")
            break
        else:
            print("Réponse invalide. Veuillez entrer 'y' ou 'n'.")

    epochs, imgsz = get_training_params()

    train_yolo(data_yaml_path, epochs, imgsz)

    upload_to_gcp("raw_data/runs_yolo")
