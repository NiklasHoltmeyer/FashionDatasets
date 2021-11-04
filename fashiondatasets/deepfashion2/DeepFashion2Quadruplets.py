import os
from pathlib import Path

from fashiondatasets.deepfashion2.helper.pairs._aggregate_collections import splits
from fashiondatasets.deepfashion2.helper.pairs.deep_fashion_2_pairs_generator import DeepFashion2PairsGenerator
from fashiondatasets.own.helper.quad_to_ds import build_pairs_ds_fn
from fashiondatasets.utils.list import parallel_map


class DeepFashion2Quadruplets:
    """
    Build DeepFashion Quadtruplets (Deterministic)
    """

    def __init__(self, base_path, format, split_suffix="", nrows=None, embedding=None):
        self.base_path = base_path
        self.split_suffix = split_suffix
        self.format = format

        self.is_triplet = (format == "triplet")
        self.nrows = nrows
        self.embedding = embedding

    def load_as_datasets(self, validate_paths=False):
        data = self.load(validate_paths=validate_paths)
        datasets = {}

        def load_x(_apnns, x):
            return list(
                map(
                    lambda apnn: apnn[x],
                    _apnns
                )
            )

        build_pairs_ds = build_pairs_ds_fn(self.is_triplet)

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
        Load quadruplets from CSV. If CSV doesnt exist, it will be generated before reading from it.
        :return:
        """

        def quadruplets_map_full_paths(_quadruplets):
            full_path = lambda split, x: os.path.join(self.base_path, split + self.split_suffix, "images", x)
            apnn_full_path = lambda split, apnn: [full_path(split, x) for x in apnn]

            map_df_full_path = lambda split, df: list(map(lambda apnn: apnn_full_path(split, apnn), df))

            return {
                split: map_df_full_path(split, apnns) for split, apnns in _quadruplets.items()
            }

        load_split = lambda split: DeepFashion2PairsGenerator.load_pairs_from_csv(base_path=self.base_path,
                                                                                  split=split,
                                                                                  nrows=self.nrows)
        unpacking_results = lambda df: df.values

        id_to_jpg = lambda x: str(x).zfill(6) + ".jpg"

        apnn_id_to_jpg = lambda apnn: list(map(id_to_jpg, apnn))
        apnn_ids_to_jps = lambda apnns: list(map(apnn_id_to_jpg, apnns))

        quadruplets_dfs = {split: load_split(split) for split in splits}
        quadruplets_unpacked = {split: unpacking_results(df) for split, df in quadruplets_dfs.items()}
        quadruplets = {split: apnn_ids_to_jps(apnns) for split, apnns in quadruplets_unpacked.items()}

        quadruplets = quadruplets_map_full_paths(quadruplets)

        if validate_paths:
            DeepFashion2Quadruplets.validate(quadruplets)

        return quadruplets

    @staticmethod
    def validate(quadruplets):
        """Just validating the Existences of all Paths. Everything else is Validated at the Generation Stage.
        """
        path_exists = lambda p: Path(p).exists()

        def validate_apnn_path(apnn):
            files_exist = all(list(map(path_exists, apnn)))
            if not files_exist:
                print(
                    f"at least one File missing [(Path, Missing), ...]! {list(zip(apnn, list(map(path_exists, apnn))))}"
                )
            return files_exist

        all_files_exist = all(
            [all(parallel_map(apnns, validate_apnn_path, desc=f"Validating {split}-Split")) for split, apnns in
             quadruplets.items()])
        assert all_files_exist


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_256"
    results = DeepFashion2Quadruplets(base_path, format="triplet").load(validate_paths=False)
    print(results)
