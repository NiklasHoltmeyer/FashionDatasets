from pathlib import Path
import pandas as pd
from tqdm.contrib.concurrent import thread_map
from utils.parallel_programming import calc_chunk_size


class Quadruplets:
    def __init__(self, BASE_PATH):
        df = self.load_as_df(BASE_PATH)
        print(df.to_dict())

    @staticmethod
    def load_as_df(base_path, **kwargs):
        df_path = Path(base_path, "quadruplet.csv")
        df = pd.read_csv(df_path, sep=";")

        df = df if not kwargs.get("map_full_paths", False) else Quadruplets._map_full_paths(df)

        if kwargs.get("validate_paths", False):
            assert Quadruplets.validate_paths(df), "Invalid Paths"

        return df

    @staticmethod
    def load_as_quadruplets(base_path):
        df = Quadruplets.load_as_df(base_path)
        return df.to_dict("results")

    @staticmethod
    def _map_full_paths(df):
        for path_key in Quadruplets.list_path_colum_keys(df):
            df[path_key] = df[path_key].map(lambda p: Path(base_path + "\\" + p))  # Path(bp, p) doesnt work on Win.
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
        "map_full_paths": True,
        "validate_paths": False
    }

    Quadruplets(base_path)
