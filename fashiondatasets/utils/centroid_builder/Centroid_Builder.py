import os
from collections import defaultdict
from pathlib import Path

import tensorflow as tf
from tqdm.auto import tqdm

from fashiondatasets.deepfashion1.helper.deep_fashion_1_pairs_generator import DeepFashion1PairsGenerator
from fashiondatasets.own.helper.mappings import preprocess_image

from fashionscrapper.utils.list import flatten, distinct
import numpy as np

assert tf is not None or True  # PyCharm removes the Imports, even tho the Function/Classes are used
assert preprocess_image is not None or True  # PyCharm removes the Imports, even tho the Function/Classes are used
assert np is not None or True


class CentroidBuilder:
    def __init__(self, pair_generator, centroids_path, model, augmentation, batch_size=64):
        self.pair_gen = pair_generator
        self.batch_size = batch_size

        self.centroids_path = Path(centroids_path)
        self.centroids_path.mkdir(exist_ok=True, parents=True)

        self.augmentation = augmentation
        self.model = model

        if not model:
            print("WARNING " * 72)
            print("Model is None. Will only build Random Pairs!")
            print("WARNING" * 72)
        elif not augmentation:
            raise Exception("Augmentation missing")

    def build_centroid(self, images, augmentation):
        map_full_path = lambda p: str((self.pair_gen.image_base_path / p).resolve())

        paths_full = map(map_full_path, images)
        paths_full = list(paths_full)

        if len(paths_full) < 1:
            return None

        images = tf.data.Dataset.from_tensor_slices(paths_full) \
            .map(preprocess_image((224, 224), augmentation=augmentation)) \
            .batch(self.batch_size, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)

        embeddings = []

        for batch in images:
            # batch_embeddings = self.model(batch)
            batch_embeddings = [np.random.rand(2048) for b in batch]
            embeddings.extend(batch_embeddings)

        embedding_center = average_vectors(embeddings)
        return embedding_center

    def load(self, split, force=False, force_hard_sampling=False, validate=False):
        pairs = self.pair_gen.load(split, force=force_hard_sampling, validate=validate)
        split_path = self.centroids_path / split
        split_path.mkdir(parents=True, exist_ok=True)
        imgs_by_id = defaultdict(lambda: [])

        distinct_imgs = distinct(flatten(pairs.values))
        retrieve_id = lambda d: d.split("/")[-2]
        for img in distinct_imgs:
            p_id = retrieve_id(img)
            imgs_by_id[p_id].append(img)

        for p_id, imgs in tqdm(imgs_by_id.items()):
            f_path = str((split_path / p_id).resolve())
            f_path_full = f_path + ".npy"

            if force or not Path(f_path_full).exists():
                f_path = str((split_path / p_id).resolve())
                centroid = self.build_centroid(imgs, lambda d: d)

                if centroid is None:
                    continue

                np.save(f_path, centroid)
        split_path = str(split_path.resolve())
        map_npy_path = lambda _id: os.path.join(split_path, _id + ".npy")

        for k in pairs.keys():
            pairs[k + "_ctl"] = pairs[k].map(lambda i: i.split("/")[-2]).map(map_npy_path)

        pairs.to_csv(Path(self.pair_gen.base_path, split + "_ctl.csv"), index=False)
        return pairs


def average_vectors(list_of_vectors, axis=0):
    return np.sum(np.array(list_of_vectors), axis=0) / len(list_of_vectors)

if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_1_256"
    pair_gen = DeepFashion1PairsGenerator(base_path, None, "_256")
    splits = ["train", "val"]
    builder = CentroidBuilder(pair_gen, r"F:\workspace\FashNets\runs\1337_resnet50_imagenet_triplet\ctl", None, augmentation=lambda d: d)
    for split in splits:
        builder.load(split, True, False)

