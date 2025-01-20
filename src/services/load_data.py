import os
import pandas as pd
import kagglehub
from pathlib import Path

DATA_PATH = "/home/codespace/.cache/kagglehub/datasets/rounakbanik/the-movies-dataset/versions/7"
def load_data(data_path: str):
        if len(os.listdir(data_path)) == 0:
           kagglehub.dataset_download("rounakbanik/the-movies-dataset")
        return os.listdir(data_path)

def load_movies(data_file: str):
    load_data(DATA_PATH)
    META_FILE_PATH = f"{DATA_PATH}/{data_file}"
    metadata = pd.read_csv(META_FILE_PATH, low_memory=False)
    print(metadata.head(3))
    return metadata


def load_data(data_path: str):
    BASE_PATH = Path().absolute()
    file_path = BASE_PATH / data_path
    metadata = pd.read_csv(file_path, low_memory=False)
    print("Data loaded!")
    return metadata