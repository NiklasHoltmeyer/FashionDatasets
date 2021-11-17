import random
import shutil
from pathlib import Path

import pandas as pd
from fashiondatasets.deepfashion1.helper.ExtractSplits import DF1_Split_Extractor, CONSUMER
from fashiondatasets.deepfashion2.helper.pairs.similar_embeddings import find_top_k
from fashionscrapper.utils.list import flatten, distinct
from tqdm.auto import tqdm
import tensorflow as tf
from fashiondatasets.own.helper.mappings import preprocess_image

import numpy as np

assert tf is not None or True  # PyCharm removes the Imports, even tho the Function/Classes are used
assert preprocess_image is not None or True  # PyCharm removes the Imports, even tho the Function/Classes are used
assert np is not None or True


class DeepFashion1PairsGenerator:
    def __init__(self,
                 base_path,
                 model,
                 image_suffix="",
                 number_possibilities=32,
                 nrows=None,
                 batch_size=64,
                 augmentation=None,
                 n_chunks=None,
                 ):
        if n_chunks is None:
            n_chunks = 1

        self.base_path = base_path
        self.model = model
        self.split_helper = DF1_Split_Extractor(self.base_path).load_helper()

        self.splits, self.cat_name_by_idxs, self.cat_idx_by_name, self.ids_by_cat_idx = [self.split_helper[k] for k in
                                                                                         ['splits', 'cat_name_by_idxs',
                                                                                          'cat_idx_by_name',
                                                                                          'ids_by_cat_idx']]
        img_folder_name = "img" + image_suffix
        self.image_base_path = Path(base_path, img_folder_name)
        self.nrows = nrows

        self.batch_size = batch_size
        self.number_possibilities = number_possibilities
        self.n_chunks = n_chunks

        self.augmentation = augmentation

        self.embedding_path = None

        if not model:
            print("WARNING " * 72)
            print("Model is None. Will only build Random Pairs!")
            print("WARNING" * 72)
        elif not augmentation:
            raise Exception("Augmentation missing")

    def load(self, split,
             force=False,
             force_hard_sampling=False,
             validate=True,
             embedding_path=None,
             overwrite_embeddings=None,
             **kwargs):
        # force only for train
        if force_hard_sampling and not self.model:
            raise Exception("Model is None. Cannot Hard Sample")
        assert split in DeepFashion1PairsGenerator.splits()
        if overwrite_embeddings and embedding_path is None:
            raise Exception("embedding_path must be set if overwrite_embeddings")

        if overwrite_embeddings:
            self.delete_path(embedding_path)

        if embedding_path:
            self.embedding_path = Path(embedding_path)
            self.embedding_path.mkdir(parents=True, exist_ok=True)

        if kwargs != {}:
            print("WARNING", "unused Parameter!")
            print(kwargs)
            print("*" * (len("WARNING unused Parameter!")))

        csv_path = Path(self.base_path, split + ".csv")
        if force or not csv_path.exists():
            anchor_positive_negative_negatives = self.build(split, validate=validate, embedding_path=embedding_path)
            quadruplets_df = pd.DataFrame(anchor_positive_negative_negatives,
                                          columns=["anchor", "positive", "negative1", "negative2"])

            quadruplets_df.to_csv(csv_path, index=False)

        return pd.read_csv(csv_path, nrows=self.nrows).sample(frac=1).reset_index(drop=True)

    @staticmethod
    def delete_path(path):
        blacklist = [".csv", ".json"]
        path = Path(path)
        path_str = str(path.resolve())

        if any(filter(lambda bl: path_str.endswith(bl), blacklist)):
            return True
        try:
            path.unlink()
            return True
        except:
            pass
        try:
            shutil.rmtree(path_str)
            return True
        except:
            return False

    def encode_paths(self, pairs, retrieve_paths_fn):
        map_full_path = lambda p: str((self.image_base_path / p).resolve())

        # encodings_keys = self.batch_encodings.keys()
        paths = (map(retrieve_paths_fn, pairs))
        paths = flatten(paths)
        paths = distinct(paths)

        # paths = filter(lambda p: p not in encodings_keys, paths)
        paths = list(paths)
        npy_full_paths = map(self.build_npy_path, paths)
        npy_full_paths = list(npy_full_paths)

        paths_with_npy_with_exist = zip(paths, npy_full_paths) # pack and check if embeddings exist
        paths_with_npy_with_exist = filter(lambda d: d[1].exists(), paths_with_npy_with_exist)
        paths_with_npy_with_not_exist = filter(lambda d: d[1].exists(), paths_with_npy_with_exist)
        paths_with_npy_with_exist = list(paths_with_npy_with_exist)
        paths_with_npy_with_not_exist = list(paths_with_npy_with_not_exist)

        paths_not_exist = map(lambda d: d[0], paths_with_npy_with_not_exist)
        paths_full_not_exist = map(map_full_path, paths_not_exist)
        paths_full_not_exist = list(paths_full_not_exist)

        if len(paths_full_not_exist) > 1:
            images = tf.data.Dataset.from_tensor_slices(paths_full_not_exist) \
                .map(preprocess_image((224, 224), augmentation=self.augmentation)) \
                .batch(self.batch_size, drop_remainder=False) \
                .prefetch(tf.data.AUTOTUNE)

        embeddings = []

        for batch in images:
            batch_embeddings = self.model(batch)
            embeddings.extend(batch_embeddings)

        assert len(embeddings) == len(paths), f"{len(embeddings)} {len(paths)}"

        batch_encodings = {}

        for p, model_embedding in zip(paths_not_exist, embeddings):
            batch_encodings[p] = model_embedding
            if self.embedding_path:
                npy_path = self.build_npy_path(p)
                np.save(npy_path, model_embedding)

        for img_path, npy_path in paths_with_npy_with_exist:
            batch_encodings[img_path] = np.load(npy_path)

        return batch_encodings

    def build_npy_path(self, img_relative_path, suffix=""):
        assert self.embedding_path
        return self.embedding_path / img_relative_path.replace(".jpg", suffix).replace("/", "_")

    @staticmethod
    def is_valid_category(force_cat_level):
        def is_valid_category(a, p):
            if force_cat_level == 0:
                return True
            if force_cat_level == 1:
                return a.split("/")[1] == p.split("/")[1]

            if force_cat_level == 2:
                return "".join((a.split("/")[1:3])) == "".join((p.split("/")[1:3]))

            raise Exception("Unknown Cat Level")

        return is_valid_category

    @staticmethod
    def walk_anchor_positive_possibilities(split_data, force_cat_level=0):
        column_keys = ["pair_id", "cat_idx", "anchor", "positive"]
        _is_valid_cat = DeepFashion1PairsGenerator.is_valid_category(force_cat_level)
        for d in split_data:
            pair_id, cat_idx, anchor_image, positive = [d[k] for k in column_keys]

            if _is_valid_cat(anchor_image, positive):
                yield pair_id, str(cat_idx), anchor_image, positive

    def walk_anchor_positive_negative_possibilities(self, anchor_positives, ids_by_cat_idx, force_cat_level=0):
        # random.sample(split_data[possible_ids[0]][CONSUMER], 1)[0]
        _is_valid_cat = DeepFashion1PairsGenerator.is_valid_category(force_cat_level)

        n_ap_pairs = len(anchor_positives)

        image_source = (["positive"] * 4 + ["anchor"] * 4) * ((n_ap_pairs // 8) + 1)  # In-Shop vs Consumer
        image_source = image_source[:n_ap_pairs]
        # 50/50 split
        assert len(image_source) == n_ap_pairs, [len(image_source), n_ap_pairs]  # <- Per AP-Pair

        for idx, ((pair_id, cat_idxs, anchor, positive), target_img_source) in tqdm(
                enumerate((zip(anchor_positives, image_source))),
                desc="Sample Possible Negative1's",
                total=n_ap_pairs):
            random_possibilities = ids_by_cat_idx[cat_idxs]
            n_samples = min((self.number_possibilities + 20), len(random_possibilities))
            possible_negatives = random.sample(random_possibilities, n_samples)

            possible_negatives = filter(lambda d: d["pair_id"] != pair_id, possible_negatives)
            possible_negatives = map(lambda d: d[target_img_source],
                                     possible_negatives)  # <- Positive only means Shop Images - for this Statement
            possible_negatives = filter(lambda i: _is_valid_cat(positive, i), possible_negatives)
            possible_negatives = list(possible_negatives)[:self.number_possibilities]

            if len(possible_negatives) > 0:
                yield pair_id, cat_idxs, anchor, positive, possible_negatives

    def walk_anchor_positive_negative_negative_possibilities(self, anchor_positive_negatives,
                                                             ids_by_cat_idx, force_cat_level=0):
        random_cat_gen = DeepFashion1PairsGenerator.random_cat_generator(ids_by_cat_idx)
        _is_valid_cat = DeepFashion1PairsGenerator.is_valid_category(force_cat_level)

        assert force_cat_level != 2, "Force Cat Level 2 is redundant for Negative 2 use 0 or 1"

        n_apn_pairs = len(anchor_positive_negatives)
        image_source = (["positive"] * 4 + ["anchor"] * 4) * ((n_apn_pairs // 8) + 1)  # In-Shop vs Consumer
        image_source = image_source[:n_apn_pairs]
        assert len(image_source) == n_apn_pairs, [len(image_source), n_apn_pairs]

        for (pair_id, ap_cat_idx, a_img, p_img, n_img), target_img_source in tqdm(zip(anchor_positive_negatives,
                                                                                      image_source),
                                                                                  desc="Sample Possible Negative2's",
                                                                                  total=n_apn_pairs):
            n_p_id = n_img.split("/")[-2]

            r_cat = random_cat_gen(ap_cat_idx)
            possible_ids = ids_by_cat_idx[r_cat]
            n_samples = min((self.number_possibilities + 20), len(possible_ids))
            possible_negative2 = random.sample(possible_ids, n_samples)
            possible_negative2 = filter(lambda d: d["pair_id"] != pair_id and d["pair_id"] != n_p_id,
                                        possible_negative2)

            possible_negative2 = map(lambda d: d[target_img_source],
                                     possible_negative2)  # Anchor only refers to Consumer Images in this Case

            possible_negative2 = filter(lambda i: not _is_valid_cat(n_img, i), possible_negative2)

            possible_negative2 = list(possible_negative2)[:self.number_possibilities]

            if len(possible_negative2) > 0:
                yield a_img, p_img, n_img, possible_negative2

    def build_anchor_positives(self, splits, force_cat_level):
        ap_possibilities_all = list(self.walk_anchor_positive_possibilities(splits, force_cat_level))
        return [(pair_id, cat_idx, anchor_image, positive)  # just build all AP pairs
                for pair_id, cat_idx, anchor_image, positive in ap_possibilities_all]

    def build_anchor_positive_negatives(self, anchor_positives, ids_by_cat_idx, force_cat_level):
        image_paths_from_pair = lambda d: [d[2], *d[-1]]
        apn_possibilities_all = list(self.walk_anchor_positive_negative_possibilities(anchor_positives,
                                                                                      ids_by_cat_idx, force_cat_level))

        apn_possibilities_chunked = np.array_split(apn_possibilities_all, self.n_chunks)
        apns = []
        is_none, not_none, len_one = 0, 0, 0

        for apn_possibilities in tqdm(apn_possibilities_chunked, desc=f"Build APN "
                                                                      f"(BS: {self.batch_size}. C: {self.n_chunks})"):
            if self.model:
                batch_encodings = self.encode_paths(apn_possibilities, image_paths_from_pair)

                for pair_id, ap_cat_idx, a_img, p_img, n_possibilities in apn_possibilities:
                    negative_embeddings = [batch_encodings[x] for x in n_possibilities]

                    if (len(negative_embeddings)) == 1:
                        negative = n_possibilities[0]
                        len_one += 1
                    elif (len(negative_embeddings)) > 1:
                        anchor_embedding = batch_encodings[a_img]
                        idx = find_top_k([anchor_embedding], negative_embeddings, most_similar=True, k=1)[0][0]
                        negative = n_possibilities[idx]
                    else:
                        is_none += 1
                        continue

                    apns.append((pair_id, ap_cat_idx, a_img, p_img, negative))
                    not_none += 1
            else:
                for pair_id, ap_cat_idx, a_img, p_img, n_possibilities in apn_possibilities:
                    # negative = random.sample(n_possibilities, 1)[0]
                    negative = n_possibilities[0]  # <- already Shuffled
                    apns.append((pair_id, ap_cat_idx, a_img, p_img, negative))
                    len_one += 1
                    not_none += 1

        return apns

    def build_anchor_positive_negative_negatives(self, anchor_positive_negatives, ids_by_cat_idx, force_cat_level):
        image_paths_from_pair = lambda d: [d[-2], *d[-1]]
        apnn_possibilities_all = list(
            self.walk_anchor_positive_negative_negative_possibilities(anchor_positive_negatives, ids_by_cat_idx,
                                                                      force_cat_level))

        apnn_possibilities_chunked = np.array_split(apnn_possibilities_all, self.n_chunks)
        apnns = []
        for apnn_possibilities in tqdm(apnn_possibilities_chunked, desc=f"Build APNN "
                                                                        f"(BS: {self.batch_size}. C: {self.n_chunks})"):
            if self.model:
                batch_encodings = self.encode_paths(apnn_possibilities, image_paths_from_pair)

                for a, p, n, n2_possibilities in apnn_possibilities:
                    negative_embedding = batch_encodings[n]
                    negative2_embeddings = [batch_encodings[x] for x in n2_possibilities]
                    if (len(negative2_embeddings)) == 1:
                        idx = 0
                    else:
                        idx = find_top_k([negative_embedding], negative2_embeddings, most_similar=True, k=1)[0][0]
                    apnn = (a, p, n, n2_possibilities[idx])
                    apnns.append(apnn)
            else:
                for a, p, n, n2_possibilities in apnn_possibilities:
                    negative2 = random.sample(n2_possibilities, 1)[0]
                    apnn = (a, p, n, negative2)
                    apnns.append(apnn)

        return apnns

    #        self.encode_paths(apnn_possibilities, image_paths_from_pair)

    #        apnns = []
    #        for a, p, n, n2_possibilities in apnn_possibilities:
    #            negative_embedding = self.encodings[n]
    #            negative2_embeddings = [self.encodings[x] for x in n2_possibilities]
    #            if (len(negative2_embeddings)) == 1:
    #                idx = 0
    #            else:
    #                idx = find_top_k([negative_embedding], negative2_embeddings, reverse=True, k=1)[0][0]
    #            apnn = (a, p, n, n2_possibilities[idx])
    #            apnns.append(apnn)

    #        return apnns

    def build(self, split, embedding_path, validate=True):
        force_cat_level = 2

        split_data, ids_by_cat_idx = self.splits[split], self.ids_by_cat_idx[split]

        anchor_positives = self.build_anchor_positives(split_data, force_cat_level)

        if validate:
            self.validate_anchor_positives(anchor_positives)

        anchor_positive_negatives = self.build_anchor_positive_negatives(anchor_positives, ids_by_cat_idx,
                                                                         force_cat_level)

        if validate:
            self.validate_anchor_positive_negatives(anchor_positive_negatives)

        if force_cat_level == 2:
            anchor_positive_negative_negatives = self.build_anchor_positive_negative_negatives(
                anchor_positive_negatives,
                ids_by_cat_idx, 1)
        else:
            anchor_positive_negative_negatives = self.build_anchor_positive_negative_negatives(
                anchor_positive_negatives,
                ids_by_cat_idx, force_cat_level)

        if validate:
            self.validate_anchor_positive_negative_negatives(anchor_positive_negative_negatives)

        total_number_possible_anchors = len(split_data)
        success_ratio = 100 * len(anchor_positive_negative_negatives) / total_number_possible_anchors
        assert success_ratio >= 88, f"{success_ratio:.2f}% < 88.00%"

        print(f"Building Pairs Success Ratio: {success_ratio}")

        return anchor_positive_negative_negatives

    @staticmethod
    def random_cat_generator(ids_by_cat_idx):
        top_level_cats = list(ids_by_cat_idx.keys())

        def __call__(_idx):
            r_cat = None
            while not r_cat:
                cat = random.sample(top_level_cats, 1)[0]
                if cat != _idx:
                    r_cat = cat
            return str(r_cat)

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
                a_cat == n1_cat, a_cat == p_cat, a_cat != n2_cat,
                n1_cat != n2_cat,
            ])

            if not pids_valid or not cats_valid:
                msg = ""
                if not pids_valid:
                    msg += "Pair-Id-Building Failed. "
                if not cats_valid:
                    msg += "Categories Failed. "
                print(msg)
                [print(x, *y) for x, y in zip(["a", "p", "n1", "n2"], [a, p, n1, n2])]
                print(a_cat, p_cat, n1_cat, n2_cat)
                assert False

    # pair_id, cat_idx, anchor_image, possibilities[idx]
    @staticmethod
    def validate_anchor_positives(anchor_positives):
        for pair_id, cat_idx, anchor_image, positive_image in tqdm(anchor_positives, desc="Validate AP's"):
            assert pair_id in anchor_image and pair_id in positive_image, \
                f"({[pair_id, cat_idx, anchor_image, positive_image]}) "

            #            assert anchor_image.split("/")[1] == positive_image.split("/")[1], \
            #                f"A and P need to be of same Category. ({[pair_id, cat_idx, anchor_image, positive_image]})"
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

            n1_cat, n2_cat = n1.split("/")[1], n2.split("/")[1]

            assert a_pid != n2_pid
            assert n1_pid != n2_pid
            assert n1_cat != n2_cat

    @staticmethod
    def splits():
        return [
            "test", "train", "val"
        ]


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_1_256"

    for split in ["val", "train", "test"]:
        generator = DeepFashion1PairsGenerator(base_path, None, "_256")
        force = split == "val"  # <- for debugging just take the smallest split lul
        df = generator.load(split, force=force)
        DeepFashion1PairsGenerator.validate_dataframe(df)
