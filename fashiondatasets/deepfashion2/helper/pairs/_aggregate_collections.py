import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from fashionscrapper.utils.io import json_load
from fashionscrapper.utils.list import distinct, flatten

from fashiondatasets.deepfashion2.helper.annotations import Annotations
from fashiondatasets.utils.list import parallel_map

splits = ["train", "validation"]  # test
load_annotation_path = lambda _base_path, _split: Path(_base_path, _split, "annos")
load_images_path = lambda _base_path, _split: Path(_base_path, _split, "images")
load_image_infos_csv_path = lambda _base_path, _split: Path(_base_path, _split, "image_info.csv")
load_pairs_complementary_path = lambda _base_path, _split: Path(_base_path, _split, "pairs_complementary.json")
load_image_quadruplets_csv_path = lambda _base_path, _split: Path(_base_path, _split, "quadruplets.csv")
load_info_path = lambda _base_path, _split: Path(_base_path, _split, "quadruplets_info.txt")


def load_aggregated_annotations(_base_path, _split, threads=None):
    csv_path = load_image_infos_csv_path(_base_path, _split)
    pairs_complementary_path = load_pairs_complementary_path(_base_path, _split)

    if not all([csv_path.exists(), pairs_complementary_path.exists()]):
        save_aggregated_annotations(_base_path, _split, force=True, threads=threads)

    complementary_cat_ids = json_load(pairs_complementary_path)
    items = pd.read_csv(csv_path)

    return {
        "complementary_cat_ids": complementary_cat_ids,
        "images_info": items
    }


def save_aggregated_annotations(_base_path, _split, force=False, threads=None):
    assert _split in splits, f'Only the following splits are supported: {splits}'
    csv_path = load_image_infos_csv_path(_base_path, _split)
    pairs_complementary_path = load_pairs_complementary_path(_base_path, _split)

    if force and csv_path.exists():
        csv_path.unlink()

    if csv_path.exists():
        return

    annotations = list_annotations_for_paring(_base_path, _split, threads)
    annotations_by_pair_id = group_annotations_by_pair_id(annotations)
    pair_id_by_category_id = group_pair_ids_by_category_id(annotations_by_pair_id)

    data = aggregate(annotations_by_pair_id, pair_id_by_category_id)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    complementary_cat_ids = build_complimentaries(df)
    with open(pairs_complementary_path, "w") as f:
        json.dump(complementary_cat_ids, f)


def aggregate(annotations_by_pair_id, pair_id_by_category_id):
    for idx, (cat_ids_in_image, pair_ids) in enumerate(pair_id_by_category_id.items()):
        n_items_in_image = len(cat_ids_in_image.split("/"))
        for pair_id in pair_ids:
            for annotation in annotations_by_pair_id[pair_id]:
                source, image_id = annotation["source"], annotation["image_id"]
                yield {
                    "categories_in_image_idx": idx,
                    "categories_in_image": cat_ids_in_image,
                    "n_items_in_image": n_items_in_image,
                    "pair_id": pair_id,
                    "source": source,
                    "image_id": str(image_id).zfill(6)
                }


def list_annotations_for_paring(_base_path, _split, threads):
    """
    List Annotations by Split
    :param _base_path: Base Path of Dataset
    :param _split: one of ["train", "test", "validation"]
    :param threads: N Threads for Multi-Threading. [Default Cpu_Count]
    :return: List of Annotations (Dicts)
    """
    assert _split in splits, f"DeepFashion only Contains the following Splits: {splits}."

    annotation_path = load_annotation_path(_base_path, _split)
    assert annotation_path.exists(), f"{annotation_path} doesnt exist."

    item_drop_keys = ["segmentation", "scale", "zoom_in", "landmarks", "bounding_box", "category_name"]

    def flatten_categories_to_id(category_ids):
        category_ids = distinct(category_ids)
        category_ids = sorted(category_ids)
        return category_ids

    #            return "/".join(category_ids)

    lst_cat_id = lambda annotation: list(map(lambda x: x["category_id"], annotation["items"]))

    def load_clean_annotation(path):
        annotation = Annotations.load(path, flatten_item_n=True)
        clean_items = lambda item_dict: {k: v for k, v in item_dict.items() if k not in item_drop_keys}
        annotation["items"] = list(map(clean_items, annotation["items"]))
        annotation["image_id"] = path.name.replace(".json", "")  # -> e.g. 000001
        annotation["category_id"] = flatten_categories_to_id(lst_cat_id(annotation))
        return annotation

    annotation_paths = Annotations.list(annotation_path)

    return parallel_map(lst=annotation_paths,
                        fn=load_clean_annotation,
                        threads=threads,
                        desc="Load Annotations")


def group_annotations_by_pair_id(annotations):
    annotations_by_pair_id = defaultdict(lambda: [])
    for annotation in annotations:
        annotations_by_pair_id[annotation["pair_id"]].append(annotation)

    assert len(annotations) == sum(map(lambda x: len(x), annotations_by_pair_id.values()))

    return dict(annotations_by_pair_id)


def group_pair_ids_by_category_id(annotations_by_pair_id):
    def aggregate_category_ids(annotation_items):
        """
            Aggregate all Ids over all Items to one ID containing all Garments present in all Images
        """
        cat_ids = map(lambda x: x["category_id"], annotation_items)
        cat_ids = distinct(flatten(list(cat_ids)))
        cat_ids = list(map(lambda x: str(x), cat_ids))
        cat_ids = "/".join(cat_ids)
        return cat_ids

    pair_id_by_category_id = defaultdict(lambda: [])
    for pair_id, items in annotations_by_pair_id.items():
        cat_id = aggregate_category_ids(items)
        pair_id_by_category_id[cat_id].append(pair_id)

    return dict(sorted(pair_id_by_category_id.items()))


def build_complimentaries(df, threads=None):
    def group_ids_by_items_in_img():
        distinct_cat_in_img = distinct(df["categories_in_image"].values)

        group_ids_by_n_items = defaultdict(lambda: [])

        for _id in distinct_cat_in_img:
            n_items = len(_id.split("/"))
            group_ids_by_n_items[n_items].append(_id)

        group_ids_by_n_items = dict(group_ids_by_n_items)
        ids_grouped = list(group_ids_by_n_items.values())
        return ids_grouped

    def filter_negative2(d):
        left_ids, right_ids = d
        for l_id in left_ids:
            if l_id in right_ids:
                return False
        return True

    def build_combinations(l_lst, r_lst):
        for l_list_item in l_lst:
            for r_list_item in r_lst:
                yield l_list_item, r_list_item

    def build_complimentaries_for_ids(ids):
        ids = map(lambda _id: _id.split("/"), ids)
        ids = list(ids)

        undo_split = lambda o: "/".join(o)
        undo_tuple = lambda d: (undo_split(d[0]), undo_split(d[1]))

        complementarities = build_combinations(ids, ids)
        complementarities = filter(filter_negative2, complementarities)
        complementarities = map(undo_tuple, complementarities)

        aggregate_complementarities = defaultdict(lambda: [])
        for _id, compliment in complementarities:
            aggregate_complementarities[_id].append(compliment)

        return dict(aggregate_complementarities)

    ids_grouped = group_ids_by_items_in_img()
    r = parallel_map(lst=ids_grouped,
                     fn=build_complimentaries_for_ids,
                     threads=threads,
                     desc="Build Complimentary Cat_IDS")

    r_flattened = {}
    for x in r:
        r_flattened.update(x)

    return r_flattened


class _DeepFashion_DF_Helper:
    def __init__(self, df):
        self.df = df
        self.by_image_id = self.init_by_image_id()
        self.by_pair_id = self.init_by_pair_id()
        self.by_cat_id = self.init_by_cat_id()
        self.by_items_in_img = self.init_by_items_in_img()
        self.image_ids = list(self.df["image_id"].values)
        # categories_in_image_idx

    def init_by_image_id(self):
        by_image_id = {}
        for r in self.to_dict():
            _id = r["image_id"]
            by_image_id[_id] = r
        return by_image_id

    def init_by_pair_id(self):
        by_pair_id = defaultdict(lambda: [])
        for r in self.to_dict():
            _id = r["pair_id"]
            by_pair_id[_id].append(r)
        return dict(by_pair_id)

    def init_by_cat_id(self):
        by_pair_id = defaultdict(lambda: [])
        for r in self.to_dict():
            _id = r["categories_in_image_idx"]
            by_pair_id[_id].append(r)
        return dict(by_pair_id)

    def init_by_items_in_img(self):
        items_in_img = defaultdict(lambda: [])
        for r in self.to_dict():
            _id = r["categories_in_image"]
            items_in_img[_id].append(r)
        return dict(items_in_img)

    def to_dict(self):
        header = list(self.df.columns.values)

        def map_row(row):
            zipped = zip(header, row)
            return {k: v for k, v in zipped}

        return list(map(map_row, self.df.values))

    def find_by_image_id(self, _id):
        df = self.df
        return df[df["image_id"] == _id]

    def get_pair_id_by_image_id(self, _id):
        row = self.find_by_image_id(self.df, _id)
        pair_ids = row["pair_id"].values
        assert len(pair_ids) == 1
        return pair_ids[0]


class DeepFashion_DF_Helper:
    def __init__(self, images_df):
        user_images = images_df[images_df["source"] == "user"]
        shop_images = images_df[images_df["source"] == "shop"]

        assert len(user_images) + len(shop_images) == len(images_df), "unexpected Source"

        self.user = _DeepFashion_DF_Helper(user_images)
        self.shop = _DeepFashion_DF_Helper(shop_images)


if __name__ == "__main__":
    base_path = f"F:\workspace\datasets\DeepFashion2 Dataset"
    for split in splits:
        load_aggregated_annotations(base_path, _split=split)
