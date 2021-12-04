import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from fashionscrapper.utils.list import distinct
from tqdm.auto import tqdm

from fashiondatasets.deepfashion1.helper.ExtractSplits import DF1_Split_Extractor
from fashiondatasets.deepfashion1.helper.cbir_helper import build_gallery, build_queries, flatten_distinct_values, \
    save_batch_encodings, flatten_gallery
from fashiondatasets.own.helper.mappings import preprocess_image
from tensorflow import keras

class DeepFashion1CBIR:
    def __init__(self, base_path, model, image_suffix="", split_keys=None, batch_size=64):
        if split_keys is None:
            split_keys = ["val", "test"]  # <- default Splits for CBIR Benckmark according to the ReadMe

        self.split_keys = split_keys

        split_helper = DF1_Split_Extractor(base_path).load_helper()
        self.splits_data, self.cat_name_by_idxs, self.cat_idx_by_name, self.ids_by_cat_idx = [split_helper[k] for k in
                                                                                              ['splits',
                                                                                               'cat_name_by_idxs',
                                                                                               'cat_idx_by_name',
                                                                                               'ids_by_cat_idx']]
        list_of_splits = [self.splits_data[k] for k in self.split_keys]
        self.gallery = build_gallery(list_of_splits)
        self.queries = build_queries(list_of_splits)

        img_folder_name = "img" + image_suffix
        self.image_base_path = Path(base_path, img_folder_name)
        self.full_path = lambda p: str(Path(self.image_base_path, p).resolve())
        self.batch_size = batch_size

        if type(model) == str:
            self.model = keras.models.load_model(model)
        else:
            self.model = model

    def bulk_embed(self, embedding_path, zip_=False):
        embedding_path = Path(embedding_path)
        embedding_path.mkdir(parents=True, exist_ok=True)

        images_paths = self.distinct_images()
        image_full_paths = list(map(self.full_path, images_paths))

        image_chunks = np.array_split(list(zip(images_paths, image_full_paths)), 10)

        for image_chunk in tqdm(image_chunks, desc="Build Encodings (Outer)"):
            img_paths, img_full_paths = list(zip(*image_chunk))
            img_paths, img_full_paths = list(img_paths), list(img_full_paths)
            assert len(img_paths) == len(img_full_paths)

            images = tf.data.Dataset.from_tensor_slices(img_full_paths) \
                .map(preprocess_image((224, 224))) \
                .batch(self.batch_size, drop_remainder=False) \
                .prefetch(tf.data.AUTOTUNE)

            embeddings = []

            for batch in tqdm(images, desc="Build Encodings (Inner)"):
                batch_embeddings = self.model(batch)
                embeddings.extend(batch_embeddings)

            assert len(embeddings) == len(img_paths), f"{len(embeddings)} {len(img_paths)}"

            batch_encodings = {}

            for p, model_embedding in zip(img_paths, embeddings):
                batch_encodings[p] = model_embedding

            save_batch_encodings(batch_encodings, embedding_path)

        assert len(os.listdir(embedding_path)) == len(images_paths)

        if not zip_:
            return embedding_path
        return shutil.make_archive(embedding_path, 'zip', embedding_path)

    def distinct_images(self):
        gallery_flattened = flatten_distinct_values(self.gallery)
        queries_flattened = flatten_distinct_values(self.queries)

        distinct_values = len(distinct(queries_flattened + gallery_flattened)) == len(gallery_flattened) + len(
            queries_flattened)
        assert distinct_values, "There should NOT be an intersection between Query and Gallery!"

        return gallery_flattened + queries_flattened

    def validate_query_image_in_gallery(self):
        q_ids, _ = flatten_gallery(self.queries.items())
        g_ids, _ = flatten_gallery(self.gallery.items())
        q_ids = sorted(q_ids)
        g_ids = sorted(g_ids)

        for q in q_ids:
            idx = g_ids.index(q)
            del g_ids[idx]

        assert len(g_ids) == 0, "Gallery contains IDS which are not in Query"


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_1_256"  #

    cbir = DeepFashion1CBIR(base_path)
