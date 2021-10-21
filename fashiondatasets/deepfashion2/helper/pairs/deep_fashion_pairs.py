from fashiondatasets.deepfashion2.helper.pairs._aggregate_collections import load_aggregated_annotations, \
    DeepFashion_DF_Helper


class DeepFashionPairsGenerator:
    def __init__(self, base_path, **kwargs):
        self.base_path = base_path
        self.threads = kwargs.get("threads", None)
        self.kwargs = kwargs

        self.df_helper = {}
        self.complementary_cat_ids = {}

    def _load(self, split):
        annotations_info = load_aggregated_annotations(self.base_path, split=split)
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
                raise Exception("#TODO 4897")  # <- Shouldnt occur

            positive = possibles_positives.pop(0)  # take first Item -> Push it to the End of the List -> Round Robin
            possibles_positives.append(positive)

            anchor_positives.append((anchor, positive))
            assert anchor is not None and positive is not None
        assert len(anchor_positives) == len(df_helper.user.image_ids)

        return anchor_positives
    def build_anchor_positive_negatives(self, split):
        """
        Negative from Same Category. 50/50 Chance of the image beeing from Shop or Consumer
        """
        anchor_positives = self.build_anchor_positives(split)
        df_helper = self.df_helper[split]

        apn = []
        for idx, (a, p) in enumerate(anchor_positives):
            cat_id = a["categories_in_image_idx"]
            pair_id = a["pair_id"]

            for _idx in range(100):  # <- number of retries
                if (idx + _idx) % 2 == 0:
                    possible_negatives = df_helper.user.by_cat_id[cat_id]
                else:
                    possible_negatives = df_helper.shop.by_cat_id[cat_id]

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
            anchor_positives) > 0.95, f"Couldnt build enough Pairs. {100 * len(apn) / len(anchor_positives):.0f}% Successfull"
        return apn
    def build_anchor_positive_negative1_negative2(self, split):
        """
        Negative1 from Same Category. 50/50 Chance of the image beeing from Shop or Consumer
        Negative2 from diffrent Category. 50/50 Chanfe of Image beeing from Shop or Consumer
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
                raise Exception("#Todo 8964654") # <- shouldnt occur

            possible_cat = complementary_cat_ids_.pop(0)
            complementary_cat_ids_.append(possible_cat)

            if idx % 2 == 0:
                possible_negatives2 = df_helper.user.by_items_in_img[possible_cat]
            else:
                possible_negatives2 = df_helper.shop.by_items_in_img[possible_cat]

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

        assert len(apnn) == len(apn), f"Couldnt build enough Pairs. {100 * len(apnn) / len(apn):.0f}% Successfull"
        self.validate_apnn(apnn)
        return apnn

    def validate_apnn(self, apnn):
        assert all([all(d) for d in apnn]), "Atleast one None in Data"

        for d in apnn:
            # checking cat_id
            a, p, n1, n2 = list(map(lambda d: d["categories_in_image_idx"], d))

            assert n2 not in [a, p, n1], f"Negative2 in APN! APN: {n2} N2 {[a, p, n1]}"
            assert a == p == n1, f"A, P, N1 should have same Category. A: {a} P: {p} N1: {n1}"

            # checking pair_id
            a, p, n1, n2 = list(map(lambda d: d["pair_id"], d))
            assert a == p, f"A and P must be of same Item! Pair-ID (A): {a} - (P): {p}"

            assert n1 not in [a, n2], "A/P and N1 shouldnt be of same Item!"
            assert n2 not in [a], "A/P and N2 shouldnt be of same Item!"

        print("Validate APNN")


if __name__ == "__main__":
    base_path = f"F:\workspace\datasets\DeepFashion2 Dataset"
    apnn = DeepFashionPairsGenerator(base_path).build_anchor_positive_negative1_negative2("train")
    print(len(apnn))
