from pathlib import Path

from fashionscrapper.utils.list import distinct
import numpy as np
from fashiondatasets.deepfashion1.helper.ExtractSplits import DF1_Split_Extractor
from fashiondatasets.deepfashion1.helper.cbir_helper import build_gallery, build_queries, flatten_distinct_values, \
    save_batch_encodings
import tensorflow as tf

from fashiondatasets.own.helper.mappings import preprocess_image

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
        self.model = model

    def bulk_embed(self, embedding_path):
        embedding_path = Path(embedding_path)
        embedding_path.mkdir(parents=True, exist_ok=True)

        images_paths = self.distinct_images()
        image_full_paths = list(map(self.full_path, images_paths))

        image_chunks = np.array_split(list(zip(images_paths, image_full_paths)), 10)

        for image_chunk in image_chunks:
            img_paths, img_full_paths = list(zip(*image_chunk))
            assert len(img_paths) == len(img_full_paths)

            images = tf.data.Dataset.from_tensor_slices(img_full_paths) \
                .map(preprocess_image((224, 224))) \
                .batch(self.batch_size, drop_remainder=False) \
                .prefetch(tf.data.AUTOTUNE)

            embeddings = []

            for batch in images:
                batch_embeddings = self.model(batch)
                embeddings.extend(batch_embeddings)

            assert len(embeddings) == len(img_paths), f"{len(embeddings)} {len(img_paths)}"

            batch_encodings = {}

            for p, model_embedding in zip(img_paths, embeddings):
                batch_encodings[p] = model_embedding

            save_batch_encodings(batch_encodings, embedding_path)

        return embedding_path

    def distinct_images(self):
        gallery_flattened = flatten_distinct_values(self.gallery)
        queries_flattened = flatten_distinct_values(self.queries)

        distinct_values = len(distinct(queries_flattened + gallery_flattened)) == len(gallery_flattened) + len(
            queries_flattened)
        assert distinct_values, "There should NOT be an intersection between Query and Gallery!"

        return gallery_flattened + queries_flattened


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_1_256"  #

    cbir = DeepFashion1CBIR(base_path)
