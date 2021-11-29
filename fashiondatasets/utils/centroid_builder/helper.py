import os
from glob import glob
from pathlib import Path
import numpy as np

from fashiondatasets.utils.mock.dev_cfg import DEV


def validate_embedding(file, embedding_shape=(2048,)):
    data = np.load(file)

    if data.size == 1:
        assert np.nan != data, f"{file} is NaN"

    assert not np.isnan(data).any(), f"{file} contains at-least one NaN!"

    assert data.shape == embedding_shape, f"Invalid Embedding Shape! Desired Shape {embedding_shape}, " \
                                          f"Found Shape {data.shape}"


def validate_embeddings(embedding_path, emb_shape=(2048,)):
    """
    Check Shape and Values of Saved Embeddings.
    """
    n_files = 0
    for root, dirs, files in os.walk(embedding_path):
        for name in files:
            if name.endswith(".npy"):
                full_path = os.path.join(root, name)
                validate_embedding(full_path)
                n_files += 1
                if "id_00028592" in name:
                    print("x")
    if DEV:
        print(f"Validated {n_files} Files.")


if __name__ == "__main__":
    embedding_path = r"F:\workspace\FashNets\runs\ctl"
    validate_embeddings(embedding_path)
    print("WUHU")
