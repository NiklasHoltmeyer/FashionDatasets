import os
from collections import defaultdict
from pathlib import Path

import tensorflow as tf

from fashiondatasets.utils.centroid_builder.helper import validate_embeddings, validate_embedding
from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from fashiondatasets.utils.mock.dev_cfg import DEV
from fashiondatasets.utils.mock.mock_augmentation import pass_trough
from fashiondatasets.utils.mock.mock_feature_extractor import SimpleCNN
from fashionscrapper.utils.io import time_logger
from tqdm.auto import tqdm
import pandas as pd
from fashiondatasets.deepfashion1.helper.deep_fashion_1_pairs_generator import DeepFashion1PairsGenerator
from fashiondatasets.own.helper.mappings import preprocess_image

from fashionscrapper.utils.list import flatten, distinct
import numpy as np

from fashiondatasets.utils.list import filter_not_exist

assert tf is not None or True  # PyCharm removes the Imports, even tho the Function/Classes are used
assert preprocess_image is not None or True  # PyCharm removes the Imports, even tho the Function/Classes are used
assert np is not None or True

logger = defaultLogger("fashion_centroid_builder")

class CentroidBuilder:
    def __init__(self, pair_generator, centroids_path, model, augmentation, batch_size=64):
        self.pair_gen = pair_generator
        self.batch_size = batch_size * 4

        self.centroids_path = Path(centroids_path)
        self.centroids_path.mkdir(exist_ok=True, parents=True)
        self.centroids_path_str = str(self.centroids_path.resolve())

        self.augmentation = augmentation
        self.model = model

        self.image_base_path_str = str(self.pair_gen.image_base_path.resolve())

        if not model:
            logger.error("WARNING " * 72)
            logger.error("Model is None. Will only build Random Pairs!")
            logger.error("WARNING" * 72)
        elif not augmentation:
            raise Exception("Augmentation missing")

    def build_centroid(self, images, map_identity=False):
        if map_identity:
            map_full_path = lambda p: p
        else:
            map_full_path = lambda p: str((self.pair_gen.image_base_path / p).resolve())

        paths = list(images)
        #        npy_full_paths = map(self.pair_gen.build_npy_path, paths)
        #        npy_full_paths = list(npy_full_paths)
        #
        #        paths_with_npy_with_exist = zip(paths, npy_full_paths)  # pack and check if embeddings exist

        ##
        npy_full_paths = map(lambda d: self.pair_gen.build_npy_path(d, suffix=".npy"), paths)
        npy_full_paths = list(npy_full_paths)

        paths_with_npy_with_exist = list(zip(paths, npy_full_paths))  # pack and check if embeddings exist

        paths_with_npy_with_not_exist = filter_not_exist(paths_with_npy_with_exist,
                                                         not_exist=True, key=lambda d: d[1], disable_output=True)
        paths_with_npy_with_exist = filter_not_exist(paths_with_npy_with_exist,
                                                     not_exist=False, key=lambda d: d[1], disable_output=True)
        ##

        #        paths_with_npy_with_exist = filter(lambda d: d[1].exists(), paths_with_npy_with_exist)
        #        paths_with_npy_with_not_exist = filter(lambda d: d[1].exists(), paths_with_npy_with_exist)
        #        paths_with_npy_with_exist = list(paths_with_npy_with_exist)
        #        paths_with_npy_with_not_exist = list(paths_with_npy_with_not_exist)

        paths_not_exist = map(lambda d: d[0], paths_with_npy_with_not_exist)
        paths_full_not_exist = map(map_full_path, paths_not_exist)
        paths_full_not_exist = list(paths_full_not_exist)

        if len(paths_full_not_exist) > 0:
            images_ds = tf.data.Dataset.from_tensor_slices(paths_full_not_exist) \
                .map(preprocess_image((224, 224), augmentation=self.augmentation)) \
                .batch(self.batch_size, drop_remainder=False) \
                .prefetch(tf.data.AUTOTUNE)
        else:
            images_ds = []
        embeddings = []

        for batch in images_ds:
            batch_embeddings = self.model(batch)
            embeddings.extend(batch_embeddings)

        for img_path, npy_path in paths_with_npy_with_exist:
            embeddings.append(np.load(npy_path))

        if len(embeddings) > 1:
            return average_vectors(embeddings)
        return embeddings[0]

#    @time_logger(name="Pair-GEN(CTL)::Load", header="Pair-Gen (CTL)", footer="Pair-Gen (CTL) [DONE]", padding_length=50,
#                 logger=defaultLogger("fashiondataset_time_logger"), log_debug=False)
    def load(self, split,
             force=False,
             force_hard_sampling=False,
             validate=False,
             overwrite_embeddings=False,
             pairs_dataframe=None,
             **kwargs):
        embedding_path = kwargs.pop("embedding_path", None)

        map_identity, pairs_dataframe = self.read_pairs_dataframe(embedding_path, force_hard_sampling, kwargs,
                                                                  overwrite_embeddings, pairs_dataframe, split,
                                                                  validate)

        split_path = self.centroids_path / split
        split_path.mkdir(parents=True, exist_ok=True)

        self.build_centroids(force, map_identity, pairs_dataframe, split_path)

        self.build_ctl_dataframe(pairs_dataframe, split_path)
        pairs_dataframe.to_csv(Path(self.pair_gen.base_path, split + "_ctl.csv"), index=False)
        return pairs_dataframe

    def build_centroids(self, force, map_identity, pairs_dataframe, split_path):
        imgs_by_id = self.group_imgs_by_id_from_dataframe(pairs_dataframe)
        for f_path, centroid in self.walk_centroids(split_path, imgs_by_id, force, map_identity):
            np.save(f_path, centroid)

    def walk_centroids(self, split_path, imgs_by_id, force, map_identity):
        for p_id, imgs in tqdm(imgs_by_id.items(), desc=f"Build Centroids (force={force}, BS={self.batch_size})"):
            f_path = str((split_path / p_id).resolve())
            f_path_full = f_path + ".npy"

            if force or not Path(f_path_full).exists():
                f_path = str((split_path / p_id).resolve())
                centroid = self.build_centroid(imgs, map_identity=map_identity)

                if centroid is not None:
                    yield f_path, centroid

    def build_ctl_dataframe(self, pairs_dataframe, split_path):
        split_path = str(split_path.resolve())
        map_npy_path = lambda _id: os.path.join(split_path, _id + ".npy")
        self.build_ctl_columns(map_npy_path, pairs_dataframe)

    def build_ctl_columns(self, map_npy_path, pairs_dataframe):
        for k in pairs_dataframe.keys():
            pairs_dataframe[k + "_ctl"] = pairs_dataframe[k].map(lambda i: i.split("/")[-2]).map(map_npy_path)
            pairs_dataframe[k] = pairs_dataframe[k].map(lambda i: self.pair_gen.build_npy_path(i, suffix=".npy"))

    def read_pairs_dataframe(self, embedding_path, force_hard_sampling, kwargs, overwrite_embeddings, pairs_dataframe,
                             split, validate):
        map_identity = pairs_dataframe is not None
        if pairs_dataframe is None:
            pairs_dataframe = self.pair_gen.load(split, force=force_hard_sampling, validate=validate,
                                                 overwrite_embeddings=overwrite_embeddings,
                                                 embedding_path=embedding_path, **kwargs)
        if kwargs.get("nrows", None):
            pairs_dataframe = pairs_dataframe.head(kwargs["nrows"])
        return map_identity, pairs_dataframe

    def group_imgs_by_id_from_dataframe(self, pairs_dataframe):
        imgs_by_id = defaultdict(lambda: [])
        distinct_imgs = distinct(flatten(pairs_dataframe.values))
        retrieve_id = lambda d: d.split("/")[-2]
        for img in distinct_imgs:
            p_id = retrieve_id(img)
            imgs_by_id[p_id].append(img)
        return imgs_by_id


def average_vectors(list_of_vectors, axis=0):
    return np.sum(np.array(list_of_vectors), axis=0) / len(list_of_vectors)


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_1_256"
    embedding_path = r"F:\workspace\FashNets\runs\BLABLABLA"

    model = SimpleCNN.build((224, 224))
    m_augmentation = pass_trough()
    assert False, "noch falsche Augmentation"
    pair_gen = DeepFashion1PairsGenerator(base_path, model, "_256",
                                          embedding_path=embedding_path, augmentation=lambda d: d)
    # splits = ["train", "val"]
    splits = ["val"]

    builder = CentroidBuilder(pair_gen, embedding_path, model, augmentation=lambda d: d)

    for split in splits:
        ctl_df = builder.load(split, True, True, overwrite_embeddings=True) # [[a, p, n1, n2, a_ctl, p_ctl, n1_ctl, n2_ctl]] (.npy paths)
        #F:\workspace\FashNets\ctl\train
        values = ctl_df.values
        logger.info(values[0])
        exit(0)

        validate_embeddings(embedding_path)
        logger.info("Validated")
        exit(0)

        for a, p, n, nn in ctl_df.values:
            print(a)
            break
