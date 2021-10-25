from fashiondatasets.deepfashion2.helper.pairs._aggregate_collections import load_aggregated_annotations, \
    DeepFashion_DF_Helper, load_image_quadruplets_csv_path, splits, load_info_path
import pandas as pd


class DeepFashionPairsGenerator:
    def __init__(self, _base_path, **kwargs):
        self.base_path = _base_path
        self.threads = kwargs.get("threads", None)
        self.kwargs = kwargs

        self.df_helper = {}
        self.complementary_cat_ids = {}

    def _load(self, split):
        annotations_info = load_aggregated_annotations(self.base_path, _split=split)
        complementary_cat_ids, images_info = annotations_info["complementary_cat_ids"], annotations_info["images_info"]

        if self.kwargs.get("shuffle", True):
            images_info = images_info.sample(frac=1)

        df_helper = DeepFashion_DF_Helper(images_info)

        self.df_helper[split] = df_helper
        self.complementary_cat_ids[split] = complementary_cat_ids

    def build_anchor_positives(self, split):
        self._load(split)
        df_helper = self.df_helper[split]

        anchor_positives = []

        for a_img_id in df_helper.user.image_ids:
            anchor = df_helper.user.by_image_id[a_img_id]
            pair_id = anchor["pair_id"]

            possibles_positives = df_helper.shop.by_pair_id[pair_id]

            if len(possibles_positives) < 1:
                raise Exception("#TODO 4897")  # <- Shouldn't occur

            positive = possibles_positives.pop(0)  # take first Item -> Push it to the End of the List -> Round Robin
            possibles_positives.append(positive)

            anchor_positives.append((anchor, positive))
            assert anchor is not None and positive is not None
        assert len(anchor_positives) == len(df_helper.user.image_ids)

        return anchor_positives

    def build_anchor_positive_negatives(self, split):
        """
        Negative from Same Category. 50/50 Chance of the image being from Shop or Consumer
        """
        anchor_positives = self.build_anchor_positives(split)
        df_helper = self.df_helper[split]

        apn = []
        for idx, (a, p) in enumerate(anchor_positives):
            cat_id = a["categories_in_image_idx"]
            pair_id = a["pair_id"]

            for _idx in range(100):  # <- number of retries
                possible_negatives = df_helper.shop.by_cat_id[cat_id]
                # if (idx + _idx) % 2 == 0:
                #    possible_negatives = df_helper.user.by_cat_id[cat_id]
                # else:
                #    possible_negatives = df_helper.shop.by_cat_id[cat_id]

                _negative = possible_negatives.pop(0)
                possible_negatives.append(_negative)
                negative_pair_id = _negative["pair_id"]

                if negative_pair_id != pair_id:
                    negative = _negative
                    break
            else:
                continue

            assert negative is not None
            apn.append((a, p, negative))
        assert len(apn) / len(
            anchor_positives) > 0.93, f"Couldn't build enough Pairs. {100 * len(apn) / len(anchor_positives):.0f}% " \
                                      f"Successful "
        return apn

    def build_anchor_positive_negative1_negative2(self, split):
        """
        Negative1 from Same Category. 50/50 Chance of the image being from Shop or Consumer
        Negative2 from different Category. 50/50 Chance of Image being from Shop or Consumer
        :param split:
        :return:
        """

        apn = self.build_anchor_positive_negatives(split)
        complementary_cat_ids = self.complementary_cat_ids[split]
        df_helper = self.df_helper[split]

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

            negative2 = None
            _a_id = anchor["categories_in_image_idx"]

            for _ in range(100):  # <- number of retries
                _negative = possible_negatives2.pop(0)
                possible_negatives2.append(_negative)
                negative_pair_id = _negative["pair_id"]
                _n_id = _negative["categories_in_image_idx"]

                if negative_pair_id != pair_id and _n_id != _a_id:
                    negative2 = _negative
                    break
            if negative2:
                apnn.append((anchor, positive, negative, negative2))

        assert len(apnn) == len(apn), f"Couldn't build enough Pairs. {100 * len(apnn) / len(apn):.0f}% Successful"
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
    base_path = f"F:\workspace\datasets\DeepFashion2 Dataset"
    for split in splits:
        apnn = DeepFashionPairsGenerator(base_path).build_anchor_positive_negative1_negative2(split)
        DeepFashionPairsGenerator.save_pairs_to_csv(base_path, split, apnn)
