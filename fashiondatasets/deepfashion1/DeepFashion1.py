from pathlib import Path

from fashionnets.models.layer.Augmentation import compose_augmentations

from fashiondatasets.deepfashion1.helper.deep_fashion_1_pairs_generator import DeepFashion1PairsGenerator
from fashiondatasets.own.helper.quad_to_ds import build_pairs_ds_fn
from tqdm.auto import tqdm

from fashiondatasets.utils.centroid_builder.Centroid_Builder import CentroidBuilder
from fashiondatasets.utils.mock.mock_augmentation import pass_trough
from fashiondatasets.utils.mock.mock_feature_extractor import SimpleCNN


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
                 n_chunks=None,
                 embedding_path=None,
                 hard_sampling=False):
        self.base_path = base_path
        self.model = model
        self.image_suffix = image_suffix
        self.number_possibilities = number_possibilities
        self.nrows = nrows
        self.batch_size = batch_size

        assert generator_type in ["ctl", "apn"]

        if hard_sampling:
            b_model = model
            assert model is not None
        else:
            b_model = None

        apn_pair_gen = DeepFashion1PairsGenerator(base_path, model=b_model,
                                                  image_suffix=image_suffix,
                                                  number_possibilities=number_possibilities,
                                                  nrows=nrows,
                                                  batch_size=batch_size,
                                                  n_chunks=n_chunks,
                                                  augmentation=augmentation,
                                                  embedding_path=embedding_path
                                                  )

        if generator_type == "apn":
            self.pair_gen = apn_pair_gen
            self.is_ctl = False
        elif generator_type == "ctl":
            self.pair_gen = CentroidBuilder(apn_pair_gen, embedding_path, model=model,
                                            augmentation=augmentation,
                                            batch_size=batch_size)
            self.is_ctl = True

    def load_split(self, split, is_triplet, force, force_hard_sampling, **kwargs):
        embedding_path = kwargs.pop("embedding_path", None)
        assert split in DeepFashion1PairsGenerator.splits()
        if self.is_ctl:
            assert embedding_path, "embedding_path Required for CTL"

        df = self.pair_gen.load(split, force=force, force_hard_sampling=force_hard_sampling,
                                embedding_path=embedding_path, **kwargs)

        cols = ['anchor', 'positive', 'negative1', 'negative2']
        img_base_path = Path(self.base_path, f"img{self.image_suffix}")

        map_full_path = lambda p: str((img_base_path / p).resolve())
        map_full_paths = lambda lst: list(map(map_full_path, lst))
        load_values = lambda c: list(map_full_paths(df[c].values))

        a, p, n1, n2 = [load_values(c) for c in tqdm(cols, f"{split}: Map full Paths")]
        assert len(a) == len(p) and len(p) == len(n1) and len(n1) == len(n2)

        is_ctl = len(df.keys()) == 8
        pair_builder = build_pairs_ds_fn(is_triplet, is_ctl)

        if is_ctl:
            cols_ctl = [x + "_ctl" for x in cols]
            ctls = [df[c].values for c in cols_ctl]
            self._build_missing_embeddings(is_triplet, a, n1, embedding_path=embedding_path,**kwargs)

            if type(embedding_path) == str:
                embedding_path_str = embedding_path
            else:
                embedding_path_str = str(embedding_path.resolve())

            embedding_path_path = Path(embedding_path)
            img_path = str(self.pair_gen.pair_gen.image_base_path.resolve())
            def inverse_path(p):
                f_name = (p.replace(embedding_path_str, "")
                          .replace("\\", "/")
                          .replace("-", "/")
                          .replace(".npy", ".jpg"))
                p = Path(img_path + "/" + f_name)
                return str(p.resolve())

            a = list(map(inverse_path, a))

            return pair_builder(a, p, n1, n2, ctls=ctls), len(a)
        else:
            return pair_builder(a, p, n1, n2), len(a)

    def load(self, is_triplet, force, force_hard_sampling, splits=None, **kwargs):
        datasets = {}
        embedding_path = kwargs.pop("embedding_path", None)
        # "test", "train", "val"
        if splits is None:
            splits = DeepFashion1PairsGenerator.splits()

        for split in splits:
            force = split == "train" and force
            force_hard_sampling = split == "train" and force_hard_sampling

            ds, n_items = self.load_split(split, is_triplet, force=force,
                                          force_hard_sampling=force_hard_sampling,
                                          embedding_path=embedding_path, **kwargs)

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

        from fashionscrapper.utils.list import distinct

        not_existing_npys = a if is_triplet else a + n1

        not_existing_npys = distinct(not_existing_npys)

        missing_embeddings = map(self.pair_gen.pair_gen.build_jpg_path, not_existing_npys)
        missing_embeddings = list(missing_embeddings)

        if type(self.pair_gen.pair_gen.image_base_path) == str:
            img_base_path_str = self.pair_gen.pair_gen.image_base_path
        else:
            img_base_path_str = str(self.pair_gen.pair_gen.image_base_path.resolve())

        def inverse_path(p):
            return p.replace(img_base_path_str, "").replace("\\", "/")


        embedding_path = str(Path(embedding_path).resolve())

        clean = lambda d: inverse_path(str(d.resolve()))
        missing_embeddings = map(clean, missing_embeddings)
        missing_embeddings = list(missing_embeddings)

        # encode_paths(missing_embeddings, retrieve_paths_fn)

        self.pair_gen.pair_gen.embedding_path = Path(embedding_path)

        self.pair_gen.pair_gen.encode_paths([missing_embeddings], retrieve_paths_fn=lambda d: d,
                                            assert_saving=True)

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

    print(datasets)
