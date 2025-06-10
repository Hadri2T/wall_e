# from ultralytics import YOLO
# import  cv2
# # Load the model
# model = YOLO('/Users/comelubrano/code/Hadri2T/wall_e/raw_data/runs_yolo/train/weights/best.pt')  # or 'last.pt' for the last checkpoint
# # Run inference on an image
# # results = model("/Users/dodohellio/code/DodooHellio/Skylight/Sandbox/Roboflow_Model/Horse-pose-estimation.v6i.yolov8/IMG_20250505_101041.jpg")
# results = model("/Users/comelubrano/code/Hadri2T/wall_e/raw_data/test_image/bouteilles-en-plastique-dans-l-océan-bouteille-211244766.webp")
# # Visualize the results
# img = results[0].plot()
# cv2.imshow('YOUPI', img)
# key = cv2.waitKey(0)

import os
from ultralytics import YOLO
import cv2

# Paramètres GCP
BUCKET_NAME = "wall-e-bucket1976"
REMOTE_MODEL_PATH = "yolo/yolov8n.pt"  # Chemin exact du fichier dans le bucket
LOCAL_MODEL_PATH = "yolov8n.pt"

# Vérifier si le modèle existe, sinon le télécharger
if not os.path.exists(LOCAL_MODEL_PATH):
    os.system(f"gcloud storage cp gs://{BUCKET_NAME}/{REMOTE_MODEL_PATH} {LOCAL_MODEL_PATH}")

# Charger le modèle YOLO
model = YOLO(LOCAL_MODEL_PATH)

# Charger une image locale pour la prédiction
image_path = "/Users/comelubrano/code/Hadri2T/wall_e/projet/Test_image/download-1.jpg"
results = model(image_path)

# Visualisation des résultats
img = results[0].plot()
cv2.imshow('YOUPI', img)
key = cv2.waitKey(0)

print(model.names)
