import os
from pathlib import Path
import pandas as pd
from fashionscrapper.utils.parallel_programming import calc_chunk_size
from tqdm.contrib.concurrent import thread_map
from tqdm.auto import tqdm
import tensorflow as tf
from random import choice
tqdm.pandas()


class Quadruplets:
    def __init__(self, BASE_PATH, **kwargs):
        self.df = self.load_as_df(BASE_PATH, **kwargs)
        self.kwargs = kwargs

    def __len__(self):
        return len(self.df)

    @staticmethod
    def load_as_df(base_path, **kwargs):
        csv_name = kwargs.get("csv_name", "quadruplet.csv")

        df_path = Path(base_path, csv_name)
        nrows = kwargs.get("nrows", None)
        df = pd.read_csv(df_path, sep=";", nrows =nrows)

        map_full_paths = kwargs.get("map_full_paths", False)
        resolve_paths = kwargs.get("resolve_paths", map_full_paths)

        df = df if not map_full_paths else Quadruplets._map_full_paths(df, base_path, resolve_paths)

        if kwargs.get("validate_paths", False):
            print("Validate Paths")
            assert Quadruplets.validate_paths(df), "Invalid Paths"

        return df

    def load_as_dataset(self):
        self.kwargs["map_full_paths"] = True

        _format = self.kwargs.get("format", "")
        assert _format in ["quadruplet", "triplet"]

        quads = self.df if self.df is not None else Quadruplets.load_as_df(base_path, **self.kwargs)

        a, p, n1, n2 = quads["a_path"].values, quads["p_path"].values, quads["n1_path"].values, quads["n2_path"].values
        a_ds, p_ds = tf.data.Dataset.from_tensor_slices(a), tf.data.Dataset.from_tensor_slices(p)

        if "triplet" in _format:
            random_choice = lambda d: choice([d[0], d[1]])

            n = zip(n1, n2)
            n = map(random_choice, n)
            n = list(n)

            n_ds = tf.data.Dataset.from_tensor_slices(n)
            return tf.data.Dataset.zip((a_ds, p_ds, n_ds))

        if "quadruplet" in _format:
            n1_ds, n2_ds = tf.data.Dataset.from_tensor_slices(n1), tf.data.Dataset.from_tensor_slices(n2)
            return tf.data.Dataset.zip((a_ds, p_ds, n1_ds, n2_ds))

        raise Exception('Unknown Format! Supported Formats {"quadruplet", "triplet"}')


    @staticmethod
    def load_as_quadruplets(base_path, **kwargs):
        df = Quadruplets.load_as_df(base_path, **kwargs)
        return df.to_dict("results")

    @staticmethod
    def _map_full_paths(df, base_path, resolve_paths, **kwargs):
        threads = kwargs.get("threads", os.cpu_count())
#        map_join_path = lambda p: os.path.join(base_path, ())   #os.path.join()
#        map_path = lambda p: str(Path(base_path + "\\" + p).resolve()) if resolve_paths else \
#            lambda p: Path(base_path + "\\" + p)
        map_path = lambda p: base_path + p
        # Path(bp, p) doesnt work on Win.

        for path_key in tqdm(Quadruplets.list_path_colum_keys(df), desc="Prepare Paths"):
            df[path_key] = df[path_key].map(map_path)
        return df

    @staticmethod
    def list_path_colum_keys(df):
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

        path_cols = Quadruplets.list_path_colum_keys(df)
        total = sum(map(lambda paths: len(df[paths]), path_cols))
        jobs = walk_paths(df, path_cols)

        chunk_size = calc_chunk_size(n_workers=8, len_iterable=total)

        r = thread_map(validate_image, jobs, max_workers=8, total=total,
                       chunksize=chunk_size, desc=f"Validate Images ({8} Threads)")

        n_successful = sum(r)
        print(f"{n_successful} / {total} Images = {100 * n_successful / total}%  Exist")
        return n_successful == total

if __name__ == "__main__":
    base_path = "F:\\workspace\\datasets\\own"

    settings = {
        "format": "triplet",
        "resolve_paths": True
    }

    ds = Quadruplets.load_as_dataset(base_path, **settings)
    print(ds)