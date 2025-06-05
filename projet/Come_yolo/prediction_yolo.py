from ultralytics import YOLO
import  cv2
# Load the model
model = YOLO('runs/detect/train3/weights/best.pt')  # or 'last.pt' for the last checkpoint
# Run inference on an image
# results = model("/Users/dodohellio/code/DodooHellio/Skylight/Sandbox/Roboflow_Model/Horse-pose-estimation.v6i.yolov8/IMG_20250505_101041.jpg")
results = model("/Users/dodohellio/Downloads/Namibie.jpg")
# Visualize the results
img = results[0].plot()
cv2.imshow('YOUPI', img)
key = cv2.waitKey(0)
   