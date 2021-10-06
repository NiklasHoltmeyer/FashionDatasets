import os
from pathlib import Path
from PIL import Image

def load_img(path):
    return Image.open(path)

def save_image(data, path):
    return Image.fromarray(data).save(path)

def list_dir_abs_path(path):
    path = Path(path)
    dirs = os.listdir(path)
    return list(map(lambda x: path/x, dirs))
