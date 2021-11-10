from pathlib import Path

from fashiondatasets.deepfashion1.helper.deep_fashion_1_pairs_generator import DeepFashion1PairsGenerator
from fashiondatasets.own.helper.quad_to_ds import build_pairs_ds_fn
from tqdm.auto import tqdm


class DeepFashion1Dataset:
    def __init__(self,
                 base_path,
                 model,
                 image_suffix="",
                 number_possibilities=32,
                 nrows=None,
                 batch_size=64,
                 n_chunks=None):
        self.base_path = base_path
        self.model = model
        self.image_suffix = image_suffix
        self.number_possibilities = number_possibilities
        self.nrows = nrows
        self.batch_size = batch_size
        self.pair_gen = DeepFashion1PairsGenerator(base_path, model=model,
                                                   image_suffix=image_suffix,
                                                   number_possibilities=number_possibilities,
                                                   nrows=nrows,
                                                   batch_size=batch_size,
                                                   n_chunks=n_chunks
                                                   )

    def load_split(self, split, is_triplet, force):
        assert split in DeepFashion1PairsGenerator.splits()

        df = self.pair_gen.load(split, force=force)

        cols = ['anchor', 'positive', 'negative1', 'negative2']
        img_base_path = Path(self.base_path, f"img{self.image_suffix}")

        map_full_path = lambda p: str((img_base_path / p).resolve())
        map_full_paths = lambda lst: list(map(map_full_path, lst))
        load_values = lambda c: list(map_full_paths(df[c].values))

        a, p, n1, n2 = [load_values(c) for c in tqdm(cols, f"{split}: Map full Paths")]
        assert len(a) == len(p) and len(p) == len(n1) and len(n1) == len(n2)

        pair_builder = build_pairs_ds_fn(is_triplet)

        return pair_builder(a, p, n1, n2), len(a)

    def load(self, is_triplet, force_train_recreate, splits=None):
        datasets = {}
        # "test", "train", "val"
        if splits is None:
            splits = DeepFashion1PairsGenerator.splits()

        for split in splits:
            force = split == "train" and force_train_recreate
            ds, n_items = self.load_split(split, is_triplet, force)

            if split == "val":
                split = "validation"

            datasets[split] = {
                "dataset": ds,
                "n_items": n_items
            }

        return datasets
