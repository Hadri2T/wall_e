from ultralytics import YOLO
import  cv2
# Load the model
model = YOLO('/Users/comelubrano/code/Hadri2T/wall_e/raw_data/runs_yolo/train/weights/best.pt')  # or 'last.pt' for the last checkpoint
# Run inference on an image
# results = model("/Users/dodohellio/code/DodooHellio/Skylight/Sandbox/Roboflow_Model/Horse-pose-estimation.v6i.yolov8/IMG_20250505_101041.jpg")
results = model("/Users/comelubrano/code/Hadri2T/wall_e/raw_data/test_image/bouteilles-en-plastique-dans-l-océan-bouteille-211244766.webp")
# Visualize the results
img = results[0].plot()
cv2.imshow('YOUPI', img)
key = cv2.waitKey(0)
