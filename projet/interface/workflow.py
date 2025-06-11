from ultralytics import YOLO


model = YOLO("raw_data/runs_yolo/train/weights/best.pt")
print(model.names)
