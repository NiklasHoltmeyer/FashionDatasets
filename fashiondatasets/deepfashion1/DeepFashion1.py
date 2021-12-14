import os
from pathlib import Path

from fashionnets.models.layer.Augmentation import compose_augmentations
from fashionscrapper.utils.io import time_logger
import numpy as np
from fashionscrapper.utils.list import distinct, flatten

from fashiondatasets.deepfashion1.helper.deep_fashion_1_pairs_generator import DeepFashion1PairsGenerator
from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.own.helper.quad_to_ds import build_pairs_ds_fn
from tqdm.auto import tqdm

from fashiondatasets.utils.centroid_builder.Centroid_Builder import CentroidBuilder
from fashiondatasets.utils.list import filter_not_exist
from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from fashiondatasets.utils.mock.mock_augmentation import pass_trough
from fashiondatasets.utils.mock.mock_feature_extractor import SimpleCNN

logger = defaultLogger("fashion_pair_gen")


class DeepFashion1Dataset:
    def __init__(self,
                 base_path,
                 model,
                 generator_type,
                 augmentation,
                 image_suffix="",
                 number_possibilities=32,
                 nrows=None,
                 batch_size=64,
                 embedding_path=None,
                 skip_build=False):

        self.base_path = base_path
        self.model = model
        self.image_suffix = image_suffix
        self.number_possibilities = number_possibilities
        self.nrows = nrows
        self.batch_size = batch_size

        assert generator_type in ["ctl", "apn"]

        apn_pair_gen = DeepFashion1PairsGenerator(base_path, model=model,
                                                  image_suffix=image_suffix,
                                                  number_possibilities=number_possibilities,
                                                  nrows=nrows,
                                                  batch_size=batch_size,
                                                  augmentation=augmentation,
                                                  embedding_path=embedding_path,
                                                  skip_build=skip_build
                                                  )

        if generator_type == "apn":
            self.pair_gen = apn_pair_gen
            self.is_ctl = False
        elif generator_type == "ctl":
            assert model, "Model required for CTL"
            self.pair_gen = CentroidBuilder(apn_pair_gen, embedding_path, model=model,
                                            augmentation=augmentation,
                                            batch_size=batch_size)
            self.is_ctl = True

    #    @time_logger(name="Load_Split", header="DeepFashion-DS", footer="DeepFashion-DS [DONE]", padding_length=50,
    #                 logger=defaultLogger("fashiondataset_time_logger"), log_debug=False)
    def load_split(self, split, is_triplet, force, force_hard_sampling, **kwargs):
        embedding_path = kwargs.pop("embedding_path", None)
        assert split in DeepFashion1PairsGenerator.splits()
        if self.is_ctl:
            assert embedding_path, "embedding_path Required for CTL"

        if kwargs.get("force_skip_map_full", False):
            map_full_path = lambda p: str(p.resolve()) if type(p) != str else p
        else:
            map_full_path = lambda p: str((img_base_path / p).resolve())

        pair_df = kwargs.pop("df", None)
        if False: #pair_df is None: #1312
            #            df = self.pair_gen.load(split, force=force, force_hard_sampling=force_hard_sampling,
            #                                    embedding_path=embedding_path, **kwargs)
            cols = ['anchor', 'positive', 'negative1', 'negative2']
            img_base_path = Path(self.base_path, f"img{self.image_suffix}")
        else:
            cols = ['a_path', 'p_path', 'n1_path', 'n2_path']
            img_base_path = Path(self.base_path)
            img_base_path_str = str(img_base_path.resolve())
            if pair_df:
                pair_df = Quadruplets._map_full_paths(pair_df, img_base_path_str, add_file_ext=True)

            if self.is_ctl:
                self.pair_gen.pair_gen.relative_paths = False
            else:
                self.pair_gen.relative_paths = False

            map_full_path = lambda p: p

        df = self.pair_gen.load(split,
                                force=force,
                                force_hard_sampling=force_hard_sampling,
                                embedding_path=embedding_path,
                                pairs_dataframe=pair_df,
                                **kwargs)

        map_full_paths = lambda lst: list(map(map_full_path, lst))
        load_values = lambda c: list(map_full_paths(df[c].values))

        print(df.keys())
        print(df.head(1))

        a, p, n1, n2 = [load_values(c) for c in cols]

        assert len(a) == len(p) and len(p) == len(n1) and len(n1) == len(n2)

        is_ctl = len(df.keys()) == 8
        pair_builder = build_pairs_ds_fn(is_triplet, is_ctl)

        if is_ctl:
            cols_ctl = [x + "_ctl" for x in cols]
            ctls = [df[c].values for c in cols_ctl]

            self._build_missing_embeddings(is_triplet, a, n1, embedding_path=embedding_path, **kwargs)

            if type(embedding_path) == str:
                embedding_path_str = embedding_path
            else:
                embedding_path_str = str(embedding_path.resolve())

            img_path = str(self.pair_gen.pair_gen.image_base_path.resolve())

            #            if self.is_ctl:
            #                self.pair_gen.pair_gen.relative_paths=False
            #            else:
            #                self.pair_gen.relative_paths = False

            def inverse_path(p):
                p = str(p.resolve())
                f_name = (p.replace(embedding_path_str, "")
                          .replace("\\", "/")
                          .replace("(-)", "/")
                          .replace(".npy", ".jpg"))
                p = Path(img_path + os.sep + f_name)
                return p

            def path_to_str(p):
                return str(p.resolve())
            if isinstance(a[0], str):
                a = list(map(lambda p: Path(p), a))

            a = list(map(inverse_path, a))

            if not isinstance(a[0], str):
                a = list(map(path_to_str, a))

            if not isinstance(p[0], str):
                p = list(map(path_to_str, p))

            if not isinstance(n1[0], str):
                n1 = list(map(path_to_str, n1))

            if not isinstance(n2[0], str):
                n2 = list(map(path_to_str, n2))

            return pair_builder(a, p, n1, n2, ctls=ctls), len(a)
        else:
            return pair_builder(a, p, n1, n2), len(a)

    #    @time_logger(name="DF-DS::Load", header="DeepFashion-DS", footer="DeepFashion-DS [DONE]", padding_length=50,
    #                 logger=defaultLogger("fashiondataset_time_logger"), log_debug=False)
    def load(self, is_triplet, force, force_hard_sampling, splits=None, **kwargs):
        datasets = {}
        embedding_path = kwargs.pop("embedding_path", None)
        # "test", "train", "val"
        if splits is None:
            splits = DeepFashion1PairsGenerator.splits()

        dataframes = kwargs.get("dataframes", None)
        if dataframes is None:
            dataframes = [None] * len(splits)

        for split, df in zip(splits, dataframes):
            # force = split == "train" and force
            # force_hard_sampling = split == "train" and force_hard_sampling

            ds, n_items = self.load_split(split, is_triplet, force=force,
                                          force_hard_sampling=force_hard_sampling,
                                          embedding_path=embedding_path, df=df, **kwargs)

            if split == "val":
                split = "validation"

            datasets[split] = {
                "dataset": ds,
                "n_items": n_items
            }

        return datasets

    def _build_missing_embeddings(self, is_triplet, a, n1, **kwargs):
        embedding_path = kwargs.get("embedding_path", None)
        assert embedding_path, "embedding_path Required for CTL"

        not_existing_npys = a if is_triplet else a + n1
        not_existing_npys = distinct(not_existing_npys)
        if isinstance(not_existing_npys[0], str):
            not_existing_npys = list(map(lambda p: Path(p), not_existing_npys))
        not_existing_npys_str = list(map(lambda d: str(d.resolve()), not_existing_npys))

        jpg_full_path = list(map(self.pair_gen.pair_gen.build_jpg_path, not_existing_npys_str))
        jpg_full_path = map(lambda p: Path(p), jpg_full_path)
        jpg_full_path = list(jpg_full_path)

        if isinstance(self.pair_gen.pair_gen.image_base_path, str):
            img_base_path_str = self.pair_gen.pair_gen.image_base_path
        else:
            img_base_path_str = str(self.pair_gen.pair_gen.image_base_path.resolve())

        def inverse_path(p):
            relative_path = p.replace(img_base_path_str, "").replace("\\", "/")
            if not relative_path[0] == ".":
                return f".{relative_path}"
            else:
                return relative_path

        embedding_path = str(Path(embedding_path).resolve())

        relative_path = lambda d: inverse_path(str(d.resolve()))

        jpg_relative_path = list(map(relative_path, jpg_full_path))

        missing_embeddings = self.filter_embeddings_missing(jpg_full_path, jpg_relative_path)

        self.pair_gen.pair_gen.embedding_path = Path(embedding_path)

        if len(missing_embeddings) < 1:
            return

        n_chunks = 9

        if len(missing_embeddings) > n_chunks:
            missing_chunked = np.array_split(missing_embeddings, n_chunks)
        else:
            missing_chunked = [missing_embeddings]

        logger.warning("_build_missing_embeddings::encode_paths")
        for chunk_missing in missing_chunked:
            self.pair_gen.pair_gen.encode_paths([chunk_missing], retrieve_paths_fn=lambda d: d,
                                                assert_saving=True, skip_filter=True)

    def filter_embeddings_missing(self, jpg_full_path, jpg_relative_path):
        jpg_full_path = list(map(lambda x: str(x.resolve()), jpg_full_path))

        npy_full_paths = list(
            map(lambda d: self.pair_gen.pair_gen.build_npy_path(d, suffix=".npy"), jpg_relative_path)
        )

        paths_with_npy_with_exist = list(
            zip(jpg_full_path, jpg_relative_path, npy_full_paths))  # pack and check if embeddings exist

        paths_with_npy_with_not_exist = filter_not_exist(paths_with_npy_with_exist,
                                                         not_exist=True, key=lambda d: d[2],
                                                         disable_output=True,
                                                         desc="Filter Missing Embeddings",
                                                         parallel=len(paths_with_npy_with_exist) > 1000)

        jpg_path_not_exist = map(lambda d: d[:2], paths_with_npy_with_not_exist)
        jpg_path_not_exist = list(jpg_path_not_exist)

        return jpg_path_not_exist


if __name__ == "__main__":
    model = SimpleCNN.build((224, 224))
    m_augmentation = pass_trough()
    base_path = 'F:\\workspace\\datasets\\deep_fashion_1_256'
    embedding_path = r"F:\workspace\FashNets\runs\BLABLABLA"

    ds_loader = DeepFashion1Dataset(base_path=base_path,
                                    image_suffix="_256",
                                    model=model,
                                    nrows=80,
                                    augmentation=compose_augmentations()(False),
                                    generator_type="ctl",
                                    embedding_path=embedding_path)

    datasets = ds_loader.load(splits=["train", "val"],
                              is_triplet=True,
                              force=False, force_hard_sampling=False, embedding_path=embedding_path)
