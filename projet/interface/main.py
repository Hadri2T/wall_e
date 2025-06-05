from ml_logic import resize_with_padding, resize_bounding_boxes, filter_images
import pandas as pd
import os
from ml_logic.preprocessor import preprocess_and_save_dataset
from inference import predict  # à intégrer plus tard
from ml_logic.data import download_from_gcp  
import pandas as pd


if __name__ == "__main__":
