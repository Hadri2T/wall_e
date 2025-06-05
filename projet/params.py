import os

BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
LOCAL_DATA_DIR = "data/preprocessed"
TARGET_SIZE = (64, 64)
