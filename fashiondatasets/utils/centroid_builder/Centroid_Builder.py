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

    def build_centroid(self, images):
        map_full_path = lambda p: str((self.pair_gen.image_base_path / p).resolve())

        paths = list(images)
        npy_full_paths = map(self.pair_gen.build_npy_path, paths)
        npy_full_paths = list(npy_full_paths)

        paths_with_npy_with_exist = zip(paths, npy_full_paths) # pack and check if embeddings exist
        paths_with_npy_with_exist = filter(lambda d: d[1].exists(), paths_with_npy_with_exist)
        paths_with_npy_with_not_exist = filter(lambda d: d[1].exists(), paths_with_npy_with_exist)
        paths_with_npy_with_exist = list(paths_with_npy_with_exist)
        paths_with_npy_with_not_exist = list(paths_with_npy_with_not_exist)

        paths_not_exist = map(lambda d: d[0], paths_with_npy_with_not_exist)
        paths_full_not_exist = map(map_full_path, paths_not_exist)
        paths_full_not_exist = list(paths_full_not_exist)

        if len(paths_full_not_exist) > 1:
            images = tf.data.Dataset.from_tensor_slices(paths_full_not_exist) \
                .map(preprocess_image((224, 224), augmentation=self.augmentation)) \
                .batch(self.batch_size, drop_remainder=False) \
                .prefetch(tf.data.AUTOTUNE)

        embeddings = []

        for batch in images:
            batch_embeddings = self.model(batch)
            embeddings.extend(batch_embeddings)

        for img_path, npy_path in paths_with_npy_with_exist:
            embeddings.append(np.load(npy_path))

        embedding_center = average_vectors(embeddings)
        return embedding_center

    def load(self, split, force=False, force_hard_sampling=False, validate=False, **kwargs):
        embedding_path = kwargs.pop("embedding_path", None)
        pairs = self.pair_gen.load(split, force=force_hard_sampling, validate=validate,
                                   overwrite_embeddings=kwargs.get("overwrite_embeddings", False),
                                   embedding_path=embedding_path)
        split_path = self.centroids_path / split
        split_path.mkdir(parents=True, exist_ok=True)

        if kwargs != {}:
            print("WARNING", "unused Parameter!")
            print(kwargs)
            print("*"*(len("WARNING unused Parameter!")))

        imgs_by_id = defaultdict(lambda: [])

        distinct_imgs = distinct(flatten(pairs.values))
        retrieve_id = lambda d: d.split("/")[-2]
        for img in distinct_imgs:
            p_id = retrieve_id(img)
            imgs_by_id[p_id].append(img)

        for p_id, imgs in tqdm(imgs_by_id.items(), desc=f"Build Centroids (force={force})"):
            f_path = str((split_path / p_id).resolve())
            f_path_full = f_path + ".npy"

            if force or not Path(f_path_full).exists():
                f_path = str((split_path / p_id).resolve())
                centroid = self.build_centroid(imgs)

                if centroid is None:
                    continue

                np.save(f_path, centroid)

        split_path = str(split_path.resolve())
        map_npy_path = lambda _id: os.path.join(split_path, _id + ".npy")

        for k in pairs.keys():
            pairs[k + "_ctl"] = pairs[k].map(lambda i: i.split("/")[-2]).map(map_npy_path)
            pairs[k] = pairs[k].map(lambda i: self.pair_gen.build_npy_path(i, suffix=".npy"))
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

