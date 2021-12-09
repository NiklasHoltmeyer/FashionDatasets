import os
from pathlib import Path
from random import choice

import pandas as pd
import tensorflow as tf
from fashiondatasets.utils.list import parallel_map
from fashiondatasets.utils.logger.defaultLogger import defaultLogger
from tqdm.auto import tqdm
tqdm.pandas()

logger = defaultLogger("fashion_pair_gen")

class Quadruplets:
    def __init__(self, base_path, **kwargs):
        #self.df = self.load_as_df(BASE_PATH, split, **kwargs)
        self.kwargs = kwargs
        self.df = None
        self.split = kwargs.get("split", None)
        self.base_path = base_path
        if not (base_path.endswith("/") or base_path.endswith("\\")):
            self.base_path += os.sep

    def __len__(self):
        return len(self.df)

    def load_as_df(self, split=None, force=False, **kwargs):
        if split is None:
            split = self.split

        assert split

        csv_name = kwargs.get("csv_name", f"{split}.csv")

        df_path = Path(self.base_path, csv_name)
        nrows = kwargs.get("nrows", None)

        if df_path.exists() and force:
            df_path.unlink()

        if not df_path.exists():
            from fashiondatasets.own.helper.build_quadruplets import build_split
            build_split(self.base_path, split)

        df = pd.read_csv(df_path, sep=";", nrows=nrows)

        map_full_paths = kwargs.get("map_full_paths", False)

        df = df if not map_full_paths else Quadruplets._map_full_paths(df, self.base_path)

        if kwargs.get("validate_paths", False):
            logger.debug("Validate Paths")
            assert Quadruplets.validate_paths(df), "Invalid Paths"

        return df

    def load_as_dataset(self, split=None):
        if split is None:
            split = self.split

        assert split

        self.kwargs["map_full_paths"] = True

        _format = self.kwargs.get("format", "")
        _split = self.kwargs.pop("split", split)
        assert _format in ["quadruplet", "triplet"]

        quads = self.df if self.df is not None else self.load_as_df(split=split, **self.kwargs)

        a, p, n1, n2 = quads["a_path"].values, quads["p_path"].values, quads["n1_path"].values, quads["n2_path"].values
        a_ds, p_ds = tf.data.Dataset.from_tensor_slices(a), tf.data.Dataset.from_tensor_slices(p)

        n_items = len(a)

        if "triplet" in _format:
            random_choice = lambda d: choice([d[0], d[1]])

            n = zip(n1, n2)
            n = map(random_choice, n)
            n = list(n)

            n_ds = tf.data.Dataset.from_tensor_slices(n)
            return n_items, tf.data.Dataset.zip((a_ds, p_ds, n_ds))

        if "quadruplet" in _format:
            n1_ds, n2_ds = tf.data.Dataset.from_tensor_slices(n1), tf.data.Dataset.from_tensor_slices(n2)
            return n_items, tf.data.Dataset.zip((a_ds, p_ds, n1_ds, n2_ds))

        raise Exception('Unknown Format! Supported Formats {"quadruplet", "triplet"}')

    @staticmethod
    def load_as_quadruplets(base_path, split, **kwargs):
        df = Quadruplets.load_as_df(base_path, split, **kwargs)
        return df.to_dict("results")

    @staticmethod
    def _map_full_paths(df, base_path, add_file_ext=True):
        def _add_file_ext(p):
            ext = os.path.splitext(p)[-1]
            if len(ext) < 1:
                return p + ".jpg"
            return p

        if os.name == "nt":
            map_path = lambda p: base_path + "/" + p
        else:
            map_path = lambda p: base_path + "/" + p.replace("\\", "/")

        # Path(bp, p) doesnt work on Win.

        for path_key in tqdm(Quadruplets.list_path_column_keys(df), desc="Prepare Paths"):
            df[path_key] = df[path_key].map(map_path)
            if add_file_ext:
                df[path_key] = df[path_key].map(_add_file_ext)
        return df

    @staticmethod
    def list_path_column_keys(df):
        return list(filter(lambda c: "path" in c, df.columns))

    @staticmethod
    def validate_paths(df):
        def walk_paths(_df, path_cols):
            for path_key in path_cols:
                for p in df[path_key]:
                    yield p

        def validate_image(p):
            if p.exists():
                return 1
            return 0

        path_cols = Quadruplets.list_path_column_keys(df)
        total = sum(map(lambda paths: len(df[paths]), path_cols))
        jobs = walk_paths(df, path_cols)

        r = parallel_map(lst=jobs,
                         fn=validate_image,
                         desc="Validate Images",
                         total=total)

        n_successful = sum(r)
        logger.debug(f"{n_successful} / {total} Images = {100 * n_successful / total}%  Exist")
        return n_successful == total


if __name__ == "__main__":
    base_path = f"F:\workspace\datasets\own"

    settings = {
        "format": "triplet",
        "resolve_paths": True,
        "base_path": base_path,
        "split": "test",
        "map_full_paths": True,
    }

    df = Quadruplets(**settings).load_as_df(force=True, map_full_paths=True)
    n_items, ds  =Quadruplets(**settings).load_as_dataset("test")
    #load_as_dataset
    print(df.head(1))
    print(len(df))
    print(list(ds.take(1))[0])
    print(n_items)