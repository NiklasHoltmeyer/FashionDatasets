import os
from pathlib import Path

from PIL import Image


def load_img(path):
    return Image.open(path)


def save_image(data, path, create_parents=False):
    if create_parents:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return Image.fromarray(data).save(path)


def list_dir_abs_path(path):
    path = Path(path)
    dirs = os.listdir(path)
    return list(map(lambda x: path / x, dirs))
