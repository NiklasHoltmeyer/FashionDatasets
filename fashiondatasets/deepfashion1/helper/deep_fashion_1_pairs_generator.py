import random
from pathlib import Path

import pandas as pd
from fashiondatasets.deepfashion1.helper.ExtractSplits import DF1_Split_Extractor, CONSUMER
from fashiondatasets.deepfashion2.helper.pairs.similar_embeddings import find_top_k
from fashionscrapper.utils.list import flatten, distinct
from tqdm.auto import tqdm


class DeepFashion1PairsGenerator:
    def __init__(self,
                 base_path,
                 model,
                 image_suffix="",
                 number_possibilities=32,
                 nrows=None):
        self.base_path = base_path
        self.model = model
        self.split_helper = DF1_Split_Extractor(self.base_path).load_helper()

        self.splits, self.cat_name_by_idxs, self.cat_idx_by_name, self.ids_by_cat_idx = [self.split_helper[k] for k in
                                                                                         ['splits', 'cat_name_by_idxs',
                                                                                          'cat_idx_by_name',
                                                                                          'ids_by_cat_idx']]
        img_folder_name = "img" + image_suffix
        self.image_base_path = Path(base_path, img_folder_name)
        self.nrows=nrows
        self.encodings = {}
        self.batch_size = 1
        self.number_possibilities = number_possibilities

    def load(self, split, force=False):
        # force only for train
        assert split in DeepFashion1PairsGenerator.splits()

        csv_path = Path(self.base_path, split + ".csv")
        if force or not csv_path.exists():
            anchor_positive_negative_negatives = self.build(split)
            quadtruplet_df = pd.DataFrame(anchor_positive_negative_negatives,
                                          columns=["anchor", "positive", "negative1", "negative2"],
                                          nrows=self.nrows)

            quadtruplet_df.to_csv(csv_path, index=False)

        return pd.read_csv(csv_path)

    def encode_paths(self, pairs, retrieve_paths_fn):
        map_full_path = lambda p: str((self.image_base_path / p).resolve())

        encodings_keys = self.encodings.keys()
        paths = (map(retrieve_paths_fn, pairs))
        paths = flatten(paths)
        paths = distinct(paths)

        paths = filter(lambda p: p not in encodings_keys, paths)
        paths = list(paths)
        paths_full = map(map_full_path, paths)
        paths_full = list(paths_full)

        if len(paths_full) < 1:
            return

        images = tf.data.Dataset.from_tensor_slices(paths_full) \
            .map(preprocess_image((224, 224))) \
            .batch(self.batch_size, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)

        embeddings = []
        for batch in tqdm(images, desc="Build-Embeddings"):
            embeddings.extend(self.model(batch))

        for p, model_embedding in zip(paths, embeddings):
            self.encodings[p] = model_embedding

    @staticmethod
    def walk_anchor_positive_possibilities(split_data):
        for pair_id, pair_data in split_data.items():
            for anchor_image in pair_data[CONSUMER]:
                cat_idx = pair_data["cat_idx"]
                possibilities = split_data[pair_id]["shop"]

                cat_name = anchor_image.split("/")[1]

                possibilities_force_same_cat = filter(lambda p: p.split("/")[1] == cat_name, possibilities)
                possibilities_force_same_cat = list(possibilities_force_same_cat)

                if len(possibilities_force_same_cat) > 0:
                    yield pair_id, cat_idx, anchor_image, possibilities_force_same_cat

    def walk_anchor_positive_negative_possibilities(self, anchor_positives, ids_by_cat_idx, split_data):
        # random.sample(split_data[possible_ids[0]][CONSUMER], 1)[0]
        for pair_id, cat_idxs, anchor, positive in anchor_positives:
            # cat1_idx, cat2_idx = cat_idxs
            cat_name = anchor.split("/")[1]
            # random_possibilities = ids_by_cat_idx[str(cat1_idx)][str(cat2_idx)]
            random_possibilities = ids_by_cat_idx[str(cat_idxs)]
            random_possibilities = filter(lambda pid: pid != pair_id, random_possibilities)
            random_possibilities = list(random_possibilities)

            n_samples = min(self.number_possibilities, len(random_possibilities))
            possible_pair_ids = random.sample(random_possibilities, n_samples + 10)
            # + 10 random number. just pick more samples, since the samples get filterd anyways
            # only n_samples Samples will be returned
            possible_images = map(lambda pid: random.sample(split_data[pid][CONSUMER], 1)[0], possible_pair_ids)
            possible_images = filter(lambda p: p.split("/")[1] == cat_name, possible_images)
            possible_images = list(possible_images)[:n_samples]

            if len(possible_images) > 0:
                yield pair_id, cat_idxs, anchor, positive, possible_images

    def walk_anchor_positive_negative_negative_possibilities(self, anchor_positive_negatives,
                                                             split_data, ids_by_cat_idx):
        random_cat_gen = DeepFashion1PairsGenerator.random_cat_generator(ids_by_cat_idx)

        for pair_id, ap_cat_idx, a_img, p_img, n_img in anchor_positive_negatives:
            # cat1, cat2 = random_cat_gen(ap_cat_idx)
            cat_name = (a_img.split("/")[1])
            cat_idx = self.cat_idx_by_name[cat_name]

            r_cat = random_cat_gen(cat_idx)
            # possible_ids = ids_by_cat_idx[cat1][cat2]
            possible_ids = ids_by_cat_idx[str(r_cat)]

            n_samples = min(self.number_possibilities, len(possible_ids))
            possible_ids = random.sample(possible_ids, n_samples + 10)
            possible_images = map(lambda pid: random.sample(split_data[pid][CONSUMER], 1)[0], possible_ids)
            possible_images = filter(lambda p: p.split("/")[1] != cat_name, possible_images)
            possible_images = list(possible_images)[:n_samples]

            yield a_img, p_img, n_img, possible_images

    def build_anchor_positives(self, splits):
        ap_possibilities = list(self.walk_anchor_positive_possibilities(splits))[:3]

        image_paths_from_pair = lambda d: [d[2], *d[-1]]
        self.encode_paths(ap_possibilities, image_paths_from_pair)

        anchor_positives = []

        for pair_id, cat_idx, anchor_image, possibilities in ap_possibilities:
            anchor_embedding = self.encodings[anchor_image]
            positive_embeddings = [self.encodings[x] for x in possibilities]

            if (len(positive_embeddings)) == 1:
                idx = 0
            else:
                idx = find_top_k([anchor_embedding], positive_embeddings, reverse=False, k=1)[0]

            ap = (pair_id, cat_idx, anchor_image, possibilities[idx])
            anchor_positives.append(ap)

        return anchor_positives

    def build_anchor_positive_negatives(self, anchor_positives, split_data, ids_by_cat_idx):
        image_paths_from_pair = lambda d: d[-1]
        apn_possibilities = list(self.walk_anchor_positive_negative_possibilities(anchor_positives,
                                                                                  ids_by_cat_idx, split_data))
        self.encode_paths(apn_possibilities, image_paths_from_pair)

        apns = []
        for pair_id, ap_cat_idx, a_img, p_img, n_possibilities in apn_possibilities:
            anchor_embedding = self.encodings[a_img]
            negative_embeddings = [self.encodings[x] for x in n_possibilities]

            if (len(negative_embeddings)) == 1:
                idx = 0
            elif (len(negative_embeddings)) > 1:
                idx = find_top_k([anchor_embedding], negative_embeddings, reverse=True, k=1)[0]
            else:
                idx = None

            if idx:
                apn = (pair_id, ap_cat_idx, a_img, p_img, n_possibilities[idx])
                apns.append(apn)
        return apns

    def build_anchor_positive_negative_negatives(self, anchor_positive_negatives, split_data, ids_by_cat_idx):
        image_paths_from_pair = lambda d: d[-1]
        apnn_possibilities = list(
            self.walk_anchor_positive_negative_negative_possibilities(anchor_positive_negatives, split_data,
                                                                      ids_by_cat_idx))
        self.encode_paths(apnn_possibilities, image_paths_from_pair)

        apnns = []
        for a, p, n, n2_possibilities in apnn_possibilities:
            negative_embedding = self.encodings[n]
            negative2_embeddings = [self.encodings[x] for x in n2_possibilities]
            if (len(negative2_embeddings)) == 1:
                idx = 0
            else:
                idx = find_top_k([negative_embedding], negative2_embeddings, reverse=True, k=1)[0]
            apnn = (a, p, n, n2_possibilities[idx])
            apnns.append(apnn)

        return apnns

    def build(self, split, validate=False):
        split_data, ids_by_cat_idx = self.splits[split], self.ids_by_cat_idx[split]

        anchor_positives = self.build_anchor_positives(split_data)

        if validate:
            self.validate_anchor_positives(anchor_positives)

        anchor_positive_negatives = self.build_anchor_positive_negatives(anchor_positives, split_data, ids_by_cat_idx)

        if validate:
            self.validate_anchor_positive_negatives(anchor_positive_negatives)

        anchor_positive_negative_negatives = self.build_anchor_positive_negative_negatives(anchor_positive_negatives,
                                                                                           split_data, ids_by_cat_idx)
        if validate:
            self.validate_anchor_positive_negative_negatives(anchor_positive_negative_negatives)

        total_number_possible_anchors = sum([len(pair_data[CONSUMER]) for pair_id, pair_data in split_data.items()])
        success_ratio = 100 * len(anchor_positive_negative_negatives) / total_number_possible_anchors
        assert success_ratio >= 88, f"{success_ratio:.2f}% < 88.00%"

        return anchor_positive_negative_negatives

    @staticmethod
    def random_cat_generator(ids_by_cat_idx):
        top_level_cats = [int(x) for x in list(ids_by_cat_idx.keys())]

        def __call__(_idx):
            r_cat = None
            while not r_cat:
                cat = random.sample(top_level_cats, 1)[0]
                if cat != _idx:
                    r_cat = cat
            return r_cat

        return __call__

    @staticmethod
    def validate_dataframe(dataframe):
        for a, p, n1, n2 in dataframe.values:
            a, p, n1, n2 = [x.split("/")[1:] for x in [a, p, n1, n2]]

            # validate pairs
            a_pid, p_pid, n1_pid, n2_pid = [x[2] for x in [a, p, n1, n2]]

            pids_valid = all([
                a_pid == p_pid, p_pid != n1_pid, p_pid != n2_pid, n1_pid != n2_pid
            ])

            # validate cats
            # a_cat, p_cat, n1_cat, n2_cat = ["/".join(x[:2]) for x in [a, p, n1, n2]]
            a_cat, p_cat, n1_cat, n2_cat = [x[0] for x in [a, p, n1, n2]]

            cats_valid = all([
                a_cat == p_cat, p_cat == n1_cat, n1_cat != n2_cat
            ])

            if not pids_valid or not cats_valid:
                msg = ""
                if not pids_valid:
                    msg += "Pair-Id-Building Failed. "
                if not cats_valid:
                    msg += "Categories Failed. "
                print(msg)
                [print(x, *y) for x, y in zip(["a", "p", "n1", "n2"], [a, p, n1, n2])]
                assert False

    # pair_id, cat_idx, anchor_image, possibilities[idx]
    @staticmethod
    def validate_anchor_positives(anchor_positives):
        for pair_id, cat_idx, anchor_image, positive_image in tqdm(anchor_positives, desc="Validate AP's"):
            assert pair_id in anchor_image and pair_id in positive_image, \
                f"({[pair_id, cat_idx, anchor_image, positive_image]}) "

            assert anchor_image.split("/")[1] == positive_image.split("/")[1], \
                f"A and P need to be of same Category. ({[pair_id, cat_idx, anchor_image, positive_image]})"
            assert anchor_image != positive_image, \
                f"A and P should be different Images. ({[pair_id, cat_idx, anchor_image, positive_image]})"

            assert pair_id in anchor_image
            assert pair_id in positive_image  #

    @staticmethod
    def validate_anchor_positive_negatives(anchor_positive_negatives):
        # validation assumes, that ap's are validated per validate_anchor_positives
        for pair_id, cat_id, anchor, positive, negative in tqdm(anchor_positive_negatives, desc="Validate APN"):
            n_tl_cat = negative.split("/")[1]
            a_tl_cat = anchor.split("/")[1]

            assert n_tl_cat == a_tl_cat, "A,P and N should be of same Category"
            assert pair_id not in negative, "Negative and A/P should not be of same Pair Id"

    @staticmethod
    def validate_anchor_positive_negative_negatives(anchor_positive_negative_negatives):
        for a, p, n1, n2 in anchor_positive_negative_negatives:
            assert a.split("/")[1] != n2.split("/")[1]

            a_pid = a.split("/")[3]
            n1_pid = n1.split("/")[3]
            n2_pid = n2.split("/")[3]

            assert a_pid != n2_pid
            assert n1_pid != n2_pid

    @staticmethod
    def splits():
        return [
            "test", "train", "val"
        ]


if __name__ == "__main__":
    import tensorflow as tf
    from fashiondatasets.own.helper.mappings import preprocess_image

    import numpy as np

    rand_emb = lambda: list(np.random.rand(5))


    class FakeEmbedder:
        def __call__(self, batch):
            return [rand_emb() for _ in range(5)]


    embedding = FakeEmbedder()
    base_path = "D:\Download\Cts"

    for split in ["val", "train", "test"]:
        generator = DeepFashion1PairsGenerator(base_path, FakeEmbedder(), "_256")
        df = generator.load(split, force=True)
        DeepFashion1PairsGenerator.validate_dataframe(df)
