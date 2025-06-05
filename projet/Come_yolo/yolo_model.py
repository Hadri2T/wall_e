from ultralytics import YOLO


def train_yolo(data_yaml_path, epochs=3):
    model=YOLO('yolov8n.pt')

    model.train(data=data_yaml_path,
              epochs = epochs,
              imgsz = 64,
              save_period = 5,
              project='/Users/comelubrano/code/Hadri2T/wall_e/raw_data/runs_yolo',
              batch = 16
    )

if __name__ == '__main__':
    train_yolo('/Users/comelubrano/code/Hadri2T/wall_e/raw_data/ocean-waste-1/data.yaml')
