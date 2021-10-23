import os
from pathlib import Path

from fashiondatasets.deepfashion2.helper.pairs._aggregate_collections import splits
from fashiondatasets.deepfashion2.helper.pairs.deep_fashion_pairs_generator import DeepFashionPairsGenerator
from fashiondatasets.utils.list import parallel_map
import tensorflow as tf


class DeepFashionQuadruplets:
    def __init__(self, base_path, format="quadtruplet", split_suffix="", nrows=None):
        self.base_path = base_path
        self.split_suffix = split_suffix
        self.format = format

        self.is_triplet = (format != "quadtruplet")
        self.nrows=nrows

    def _build_pairs_ds_fn(self):
        """
        :param is_triplet: Triplet_loss, else Quad.
        :return: Zipped Dataframe Consisting of A, P, N or A, P, N1, N2 depending on is_triplet Flag
        """
        def zip_triplets(a, p, n):
            a_ds = tf.data.Dataset.from_tensor_slices(a)
            p_ds = tf.data.Dataset.from_tensor_slices(p)
            n_ds = tf.data.Dataset.from_tensor_slices(n)

            return tf.data.Dataset.zip((a_ds, p_ds, n_ds))

        def zip_quadruplets(a, p, n1, n2):
            a_ds = tf.data.Dataset.from_tensor_slices(a)
            p_ds = tf.data.Dataset.from_tensor_slices(p)
            n1_ds = tf.data.Dataset.from_tensor_slices(n1)
            n2_ds = tf.data.Dataset.from_tensor_slices(n2)

            return tf.data.Dataset.zip((a_ds, p_ds, n1_ds, n2_ds))

        def apnn_pairs(a, p, n1, n2):
            return zip_quadruplets(a, p, n1, n2)

        def apn_pairs(a, p, n1, n2):
            n = []
            for i, (n1, n2) in enumerate(zip(n1, n2)):
                if i % 2 == 0:
                    n.append(n1)
                else:
                    n.append(n2)
            return zip_triplets(a, p, n)

        if self.is_triplet:
            return apn_pairs
        return apnn_pairs

    def load_as_datasets(self, validate_paths=False):
        data = self.load(validate_paths=validate_paths)
        datasets = {}

        def load_x(apnns, x):
            return list(
                map(
                    lambda apnn: apnn[x],
                    apnns
                )
            )

        build_pairs_ds = self._build_pairs_ds_fn()

        for split, apnns in data.items():
            a, p, n1, n2 = list(map(lambda x: load_x(apnns, x), range(4)))
            assert len(a) == len(p) and len(p) == len(n1) and len(n1) == len(n2)

            ds = build_pairs_ds(a, p, n1, n2)

            datasets[split] = {
                "dataset": ds,
                "n_items": len(apnns)
            }

        return datasets

    def load(self, validate_paths=False):
        """
        Load Quadtruplets from CSV. If CSV doesnt exist, it will be generated before reading from it.
        :return:
        """

        def quadtruplets_map_full_paths(quadtruplets):
            full_path = lambda split, x: os.path.join(self.base_path, split + self.split_suffix, "images", x)
            apnn_full_path = lambda split, apnn: [full_path(split, x) for x in apnn]

            map_df_full_path = lambda split, df: list(map(lambda apnn: apnn_full_path(split, apnn), df))

            return {
                split: map_df_full_path(split, apnns) for split, apnns in quadtruplets.items()
            }

        load_split = lambda split: DeepFashionPairsGenerator.load_pairs_from_csv(base_path=self.base_path,
                                                                                 split=split,
                                                                                 nrows=self.nrows)
        unpacking_results = lambda df: df.values

        id_to_jpg = lambda x: str(x).zfill(6) + ".jpg"

        apnn_id_to_jpg = lambda apnn: list(map(id_to_jpg, apnn))
        apnn_ids_to_jps = lambda apnns: list(map(apnn_id_to_jpg, apnns))

        quadtruplets_dfs = {split: load_split(split) for split in splits}
        quadtruplets_unpacked = {split: unpacking_results(df) for split, df in quadtruplets_dfs.items()}
        quadtruplets = {split: apnn_ids_to_jps(apnns) for split, apnns in quadtruplets_unpacked.items()}

        quadtruplets = quadtruplets_map_full_paths(quadtruplets)

        if validate_paths:
            DeepFashionQuadruplets.validate(quadtruplets)

        return quadtruplets

    @staticmethod
    def validate(quadtruplets):
        """Just validating the Existences of all Paths. Everything else is Validated at the Generation Stage.
        """
        path_exists = lambda p: Path(p).exists()

        def validate_apnn_path(apnn):
            files_exist = all(list(map(path_exists, apnn)))
            if not files_exist:
                print(
                    f"atleast one File missing [(Path, Missing), ...]! {list(zip(apnn, list(map(path_exists, apnn))))}")
            return files_exist

        all_files_exist = all(
            [all(parallel_map(apnns, validate_apnn_path, desc=f"Validating {split}-Split")) for split, apnns in
             quadtruplets.items()])
        assert all_files_exist


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\DeepFashion2 Dataset"
    results = DeepFashionQuadruplets(base_path).load(validate_paths=True)
    print(results.keys())
