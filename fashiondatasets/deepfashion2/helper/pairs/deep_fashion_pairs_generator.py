import os
import random

from fashionscrapper.utils.list import flatten, distinct

from fashiondatasets.deepfashion2.helper.pairs._aggregate_collections import load_aggregated_annotations, \
    DeepFashion_DF_Helper, load_image_quadruplets_csv_path, splits, load_info_path
import pandas as pd
from tqdm.auto import tqdm
import tensorflow as tf

from fashiondatasets.deepfashion2.helper.pairs.similar_embeddings import find_top_k
from fashiondatasets.own.helper.mappings import preprocess_image


class DeepFashionPairsGenerator:
    def __init__(self, _base_path, embedding, number_possibilites=32, split_suffix="", **kwargs):
        self.base_path = _base_path
        self.threads = kwargs.get("threads", None)
        self.kwargs = kwargs

        self.df_helper = {}
        self.complementary_cat_ids = {}

        self.embedding = embedding
        self.split_suffix = split_suffix
        self.embedding_storage = None

        self.number_possibilites = number_possibilites

    def full_image_path(self, split, x):
        return os.path.join(self.base_path, split + self.split_suffix, "images", str(x).zfill(6) + ".jpg")

    def _load(self, split):
        annotations_info = load_aggregated_annotations(self.base_path, _split=split)
        complementary_cat_ids, images_info = annotations_info["complementary_cat_ids"], annotations_info["images_info"]

        if self.kwargs.get("shuffle", True):
            images_info = images_info.sample(frac=1)

        df_helper = DeepFashion_DF_Helper(images_info)

        self.df_helper[split] = df_helper
        self.complementary_cat_ids[split] = complementary_cat_ids

    def yield_anchor_positive_possibilites(self, df_helper):
        for a_img_id in df_helper.user.image_ids:
            anchor = df_helper.user.by_image_id[a_img_id]
            pair_id = anchor["pair_id"]

            possibles_positives = df_helper.shop.by_pair_id[pair_id]
            yield anchor, possibles_positives

    def choose_possibility_by_round_robin(self, possibilities):
        for i in range(100):  # <- 100 retries
            choice = possibilities.pop(0)  # take first Item -> Push it to the End of the List -> Round Robin
            possibilities.append(choice)
            return choice

    def choose_possibility_by_distance(self, pivot, possibilities, reverse):
        possibilites_embeddings = [x["embedding"] for x in possibilities]
        pivot_embedding = [pivot["embedding"]]

        idx = find_top_k(pivot_embedding, possibilites_embeddings, reverse=reverse, k=1)[0]
        return possibilities[idx]

    def choose_possibility(self, pivot, possibilities, reverse):
        if not self.embedding or len(possibilities) == 1:
            return self.choose_possibility_by_round_robin(possibilities)
        return self.choose_possibility_by_distance(pivot, possibilities, reverse=reverse)

    def build_anchor_positives(self, split):
        self._load(split)
        df_helper = self.df_helper[split]

        image_paths = []
        anchor_possible_positives = list(self.yield_anchor_positive_possibilites(df_helper))

        for anchor, possibles_positives in self.yield_anchor_positive_possibilites(df_helper):
            i_path = self.full_image_path(split, anchor["image_id"])
            image_paths.append(i_path)

        image_ds = tf.data.Dataset.from_tensor_slices(list(image_paths)) \
            .map(preprocess_image((224, 224))) \
            .batch(64, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)

        embeddings = []
        for batch in tqdm(image_ds, desc="Anchor-Embeddings"):
            embeddings.extend(self.embedding(batch))

        assert len(embeddings) == len(image_paths)

        anchor_possible_positives_embeddings = [(anchor, possibles_positives, embedding) for
                                                ((anchor, possibles_positives), embedding) in
                                                zip(anchor_possible_positives, embeddings)]

        possible_positive_images = flatten([x[1] for x in anchor_possible_positives_embeddings])
        possible_positive_images_ids = distinct([x["image_id"] for x in possible_positive_images])
        possible_positive_images = [self.full_image_path(split, x) for x in possible_positive_images_ids]

        pp_image_ds = tf.data.Dataset.from_tensor_slices(list(possible_positive_images)) \
            .map(preprocess_image((224, 224))) \
            .batch(128, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)

        pp_embeddings = []
        for batch in tqdm(pp_image_ds, desc="Positives Embeddings"):
            pp_embeddings.extend(self.embedding(batch))

        assert len(pp_embeddings) == len(possible_positive_images)

        self.embedding_storage = {id: embedding for id, embedding in zip(possible_positive_images_ids, pp_embeddings)}

        anchor_positives = []
        for anchor, possibles_positives, embedding in anchor_possible_positives_embeddings:
            anchor["embedding"] = embedding
            for pp in possibles_positives:
                pp["embedding"] = self.embedding_storage[pp["image_id"]]
            positive = self.choose_possibility(anchor, possibles_positives, reverse=False)
            anchor_positives.append((anchor, positive))

        assert len(anchor_positives) == len(anchor_possible_positives)

        return anchor_positives

    def build_anchor_positive_negatives(self, split):
        """
        Negative from Same Category. 50/50 Chance of the image being from Shop or Consumer
        """
        anchor_positives = self.build_anchor_positives(split)
        df_helper = self.df_helper[split]
        anchor_positives_negative_possibilities = []
        for idx, (a, p) in tqdm(enumerate(anchor_positives), desc="APN", total=len(anchor_positives)):
            cat_id = a["categories_in_image_idx"]
            pair_id = a["pair_id"]
            possible_negatives = df_helper.shop.by_cat_id[cat_id]
            possible_negatives = list(filter(lambda d: pair_id != d["pair_id"], possible_negatives))
            possible_negatives = random.sample(possible_negatives,
                                               min(self.number_possibilites, len(possible_negatives)))
            anchor_positives_negative_possibilities.append((a, p, possible_negatives))

        embedding_jobs = []
        for a, p, possible_negatives in anchor_positives_negative_possibilities:
            image_ids = (map(lambda d: d["image_id"], possible_negatives))
            image_ids = list(filter(lambda id: id not in self.embedding_storage.keys(), image_ids))
            embedding_jobs.extend(image_ids)

        if len(embedding_jobs) > 0:
            image_paths = list(map(lambda x: self.full_image_path(split, x), embedding_jobs))
            negative_1_ds = tf.data.Dataset.from_tensor_slices(image_paths) \
                .map(preprocess_image((224, 224))) \
                .batch(64, drop_remainder=False) \
                .prefetch(tf.data.AUTOTUNE)

            embeddings = []
            for batch in tqdm(negative_1_ds, desc="N2-Embeddings"):
                embeddings.extend(self.embedding(batch))

            self.embedding_storage.update({id: embedding for id, embedding in
                                           zip(embedding_jobs, embeddings)})

            # 1312
            raise Exception("TODO Embedd new Negatives")
            # if this Exp. occours embedd all Images from withing Embedding_jobs and append them to
            # self::embedding_storage
        ###
        apn = []
        for a, p, possible_negatives in anchor_positives_negative_possibilities:
            for pp in possible_negatives:
                pp["embedding"] = self.embedding_storage[pp["image_id"]]
            if len(possible_negatives) > 1:
                negative = self.choose_possibility(a, possible_negatives, reverse=True)
            else:
                negative = possible_negatives
            apn.append((a, p, negative))
        return apn

    def build_anchor_positive_negative1_negative2(self, split, validate=False):
        """
        Negative1 from Same Category. 50/50 Chance of the image being from Shop or Consumer
        Negative2 from different Category. 50/50 Chance of Image being from Shop or Consumer
        :param split:
        :return:
        """

        apn = self.build_anchor_positive_negatives(split)
        complementary_cat_ids = self.complementary_cat_ids[split]
        df_helper = self.df_helper[split]
        apn_possiblen2 = []

        def _complementary_cat_ids_(cat_id, depth=0):
            if depth > 0:
                _cat_id = "/".join(cat_id.split("/")[:-depth])
                if len(_cat_id) < 1:
                    return None
            else:
                _cat_id = cat_id

            cat_ids = complementary_cat_ids.get(_cat_id, None)
            if cat_ids:
                return cat_ids
            return _complementary_cat_ids_(cat_id, depth + 1)

        apnn = []

        for idx, (anchor, positive, negative) in enumerate(apn):
            cat_id = anchor["categories_in_image"]
            pair_id = anchor["pair_id"]
            complementary_cat_ids_ = _complementary_cat_ids_(cat_id, 0)

            if complementary_cat_ids_ is None:
                continue

            if len(complementary_cat_ids_) < 1:
                raise Exception("#Todo 8964654")  # <- shouldn't occur

            possible_cat = complementary_cat_ids_.pop(0)
            complementary_cat_ids_.append(possible_cat)

            #            if idx % 2 == 0:
            #                possible_negatives2 = df_helper.user.by_items_in_img[possible_cat]
            #            else:
            #                possible_negatives2 = df_helper.shop.by_items_in_img[possible_cat]
            possible_negatives2 = df_helper.user.by_items_in_img[possible_cat]

            if len(possible_negatives2) < 1:
                raise Exception("#Todo #213213")

            _a_id = anchor["categories_in_image_idx"]

            possible_negatives2 = filter(
                lambda d: d["categories_in_image_idx"] != pair_id and d["categories_in_image_idx"] != _a_id,
                possible_negatives2)
            possible_negatives2 = list(possible_negatives2)
            possible_negatives2 = random.sample(possible_negatives2,
                                                min(self.number_possibilites, len(possible_negatives2)))
            apn_possiblen2.append((anchor, positive, negative, possible_negatives2))

        #
        possible_negatives_2 = distinct(flatten([[y["image_id"] for y in x[3]] for x in apn_possiblen2]))
        image_ids = list(filter(lambda id: id not in self.embedding_storage.keys(), possible_negatives_2))
        image_paths = list(map(lambda x: self.full_image_path(split, x), image_ids))

        negative_2_ds = tf.data.Dataset.from_tensor_slices(image_paths) \
            .map(preprocess_image((224, 224))) \
            .batch(64, drop_remainder=False) \
            .prefetch(tf.data.AUTOTUNE)

        embeddings = []
        for batch in tqdm(negative_2_ds, desc="N2-Embeddings"):
            embeddings.extend(self.embedding(batch))

        self.embedding_storage.update({id: embedding for id, embedding in zip(image_ids, embeddings)})
        apnn = []
        for anchor, positive, negative, possible_negatives2 in apn_possiblen2:
            if len(negative) < 1:
                continue
            for pp in possible_negatives2:
                pp["embedding"] = self.embedding_storage[pp["image_id"]]

            if len(possible_negatives2) > 1:
                if type(negative) == list and len(negative) == 1:
                    negative = negative[0]
                negative2 = self.choose_possibility(negative, possible_negatives2, reverse=True)
            else:
                negative2 = possible_negatives2
            apnn.append((anchor, positive, negative, negative2))

        assert len(apnn) < len(apn) and len(apnn) / len(apn) > 0.9, f"Couldn't build enough Pairs. {100 * len(apnn) / len(apn):.0f}% Successful"
        if validate:
            self.validate_apnn(apnn, split)
        return apnn

    def validate_apnn(self, apnn, split):

        assert all([all(d) for d in apnn]), "At least one None in Data"
        data_sources = {"a": {"user": 0, "shop": 0}, "p": {"user": 0, "shop": 0}, "n1": {"user": 0, "shop": 0},
                        "n2": {"user": 0, "shop": 0}}
        for d in apnn:
            # checking cat_id
            a_cid, p_cid, n1_cid, n2_cid = list(map(lambda d: d["categories_in_image_idx"], d))

            assert n2_cid not in [a_cid, p_cid, n1_cid], f"Negative2 in APN! APN: {n2_cid} N2 {[a_cid, p_cid, n1_cid]}"
            assert a_cid == p_cid == n1_cid, f"A, P, N1 should have same Category. A: {a_cid} P: {p_cid} N1: {n1_cid}"

            # checking pair_id
            a_pid, p_id, n1_pid, n2_pid = list(map(lambda d: d["pair_id"], d))
            assert a_pid == p_id, f"A and P must be of same Item! Pair-ID (A): {a_pid} - (P): {p_id}"

            assert n1_pid not in [a_pid, n2_pid], "A/P and N1 shouldn't be of same Item!"
            assert n2_pid not in [a_pid], "A/P and N2 shouldn't be of same Item!"

            a_source, p_source, n1_sourced, n2_source = list(map(lambda d: d["source"], d))
            data_sources["a"][a_source] += 1
            data_sources["p"][p_source] += 1
            data_sources["n1"][n1_sourced] += 1
            data_sources["n2"][n2_source] += 1

        print(f"Validate APNN ({len(apnn)} Pairs) Consisting:")
        z_fill_length = len(f"{len(apnn)}")
        info_txt = load_info_path(self.base_path, split)

        lines = []
        for item, _dict in data_sources.items():
            total = _dict['user'] + _dict['shop']
            user_ratio, shop_ratio = _dict['user'] / total, _dict['shop'] / total
            user_ratio, shop_ratio = 100 * user_ratio, 100 * shop_ratio
            user_ratio, shop_ratio = f"{user_ratio: .0f}%", f"{shop_ratio: .0f}%"
            user_ratio = (" " * (5 - len(user_ratio))) + user_ratio  # <- padding
            shop_ratio = (" " * (5 - len(shop_ratio))) + shop_ratio
            ratio = f"{user_ratio} User-Images. {shop_ratio} In-Shop Images."

            line = (item + " " f"\t{str(_dict['user']).zfill(z_fill_length)} User " +
                    f"and {str(_dict['shop']).zfill(z_fill_length)} Shop Images."
                    + " " + ratio)
            lines.append(line + "\n")
        print(f"Write Infos to: {info_txt}")
        with open(info_txt, "w+") as f:
            f.writelines(lines)

    @staticmethod
    def pair_only_keep_image_id(apnn):
        only_keep_img_id = lambda i: str(i["image_id"]).zfill(6)
        only_img_id_over_pairs = lambda p: list(map(only_keep_img_id, p))
        return list(map(only_img_id_over_pairs, apnn))

    @staticmethod
    def save_pairs_to_csv(base_path, split, apnn):
        apnn_ids = DeepFashionPairsGenerator.pair_only_keep_image_id(apnn)
        if len(apnn_ids[0]) == 4:
            header = ["a", "p", "n1", "n2"]
        elif len(apnn_ids[0]) < 4:
            header = ["a", "p", "n"][:len(apnn_ids[0])]
        else:
            raise Exception(f"Pairs consisting of {len(apnn_ids[0])} Items not Supported.")
        quadruplets_csv_path = load_image_quadruplets_csv_path(base_path, split)
        df = pd.DataFrame(apnn_ids, columns=header)
        df.to_csv(quadruplets_csv_path, index=False)

    @staticmethod
    def load_pairs_from_csv(base_path, split, force=False, nrows=None):
        quadruplets_csv_path = load_image_quadruplets_csv_path(base_path, split)

        if force and quadruplets_csv_path.exists():
            quadruplets_csv_path.unlink()

        if not quadruplets_csv_path.exists():
            apnn = DeepFashionPairsGenerator(base_path).build_anchor_positive_negative1_negative2(split)
            DeepFashionPairsGenerator.save_pairs_to_csv(base_path, split, apnn)

        return pd.read_csv(quadruplets_csv_path, nrows=nrows)


if __name__ == "__main__":
    base_path = f"F:\workspace\datasets\deep_fashion_256"
    print(splits)
    # for split in splits: #"train"
    # apnn = DeepFashionPairsGenerator(base_path).build_anchor_positive_negative1_negative2(split)
    # DeepFashionPairsGenerator.save_pairs_to_csv(base_path, split, apnn)
    # DeepFashionPairsGenerator(base_path).build_anchor_positive_negative1_negative2(splits[1])
