# TODO Decoupling from F.Scrapper

from random import shuffle

import numpy as np
import pandas as pd
from fashionscrapper.brand.asos.consts.parser import (CATEGORIES as ASOS_CATEGORIES, BASE_PATH as ASOS_BASE_PATH)
from fashionscrapper.brand.asos.helper.database.dbhelper import list_dbs_by_category
from fashionscrapper.brand.asos.helper.download.AsosPaths import AsosPaths
from fashionscrapper.brand.asos.webelements.consts.Asos_Selectors import Asos_Selectors
from fashionscrapper.brand.hm.consts.parser import (CATEGORIES as HM_CATEGORIES, BASE_PATH as HM_BASE_PATH)
from fashionscrapper.brand.hm.helper.download.HMPaths import HMPaths
from fashionscrapper.brand.hm.webelements.consts.HM_Selectors import HM_Selectors
from fashionscrapper.brand.mango.consts.parser import (BASE_PATH as MANGO_BASE_PATH)
from fashionscrapper.utils.io import walk_entries, Json_DB
from fashionscrapper.utils.list import flatten, idx_self_reference, distinct_list_of_dicts
from fashionscrapper.brand.mango.helper.database.filter import filtered_entries


def as_df(triplets):
    anchors = [x["anchor"] for x in triplets]
    positives = [x["positive"] for x in triplets]
    negatives = [x["negative"] for x in triplets]

    triplet_df = pd.DataFrame()

    triplet_df["a_id"] = [x["id"] for x in anchors]
    triplet_df["a_view"] = [x["view"] for x in anchors]
    triplet_df["a_path"] = [x["path"] for x in anchors]

    triplet_df["p_id"] = [x["id"] for x in positives]
    triplet_df["p_view"] = [x["view"] for x in positives]
    triplet_df["p_path"] = [x["path"] for x in positives]

    triplet_df["n_id"] = [x["id"] for x in negatives]
    triplet_df["n_view"] = [x["view"] for x in negatives]
    triplet_df["n_path"] = [x["path"] for x in negatives]
    return triplet_df


def _random_references(num_idxs, max_retries=150):
    random_idxs = list(range(num_idxs))
    for _ in range(max_retries):
        shuffle(random_idxs)  # in place
        if not idx_self_reference(random_idxs):
            break
    return np.array(random_idxs)  # copy -> None otherwise


def preprocess_negative_entries(anchor_items):
    negative_items = np.array(anchor_items)
    shuffle(negative_items)

    def __clean_entry(entry):
        return [{**x, 'id': entry["id"]} for x in entry["images"]]

    assert len(anchor_items) == len(negative_items)
    negative_items = [__clean_entry(x) for x in negative_items]
    negative_flat = flatten(negative_items)

    shuffle(negative_flat)

    return negative_flat


def build_positive_anchors(entry):
    images_anchor, anchor_id = entry["images"], entry["id"]
    positive_idxs = _random_references(len(images_anchor))
    images_positives = [images_anchor[idx] for idx in positive_idxs]
    if len(positive_idxs) < 2:
        return None
    return [{"anchor": {**a, "id": anchor_id}, "positive": {**p, "id": anchor_id}} for (a, p) in
            zip(images_anchor, images_positives)]


def preprocess_positive_entries(entries):
    a_p = [build_positive_anchors(entry) for entry in entries]
    a_p = [x for x in a_p if x]
    return flatten(a_p)


def build_triplets(entries):
    negative_items = preprocess_negative_entries(entries)
    positive_items = preprocess_positive_entries(entries)

    data = []
    for row in positive_items:
        anchor_id, positive_id = row["anchor"]["id"], row["positive"]["id"]
        negative_anchor = None

        while not negative_anchor:
            possible_n_anchor = negative_items.pop()
            if possible_n_anchor["id"] != anchor_id:
                negative_anchor = possible_n_anchor
            elif len(negative_items) < 2:  # for the random chance of only the same id being left
                break
            else:
                negative_items.insert(0, possible_n_anchor)

        data.append({**row, "negative": negative_anchor})
    validate_triplets(data)
    return data


def validate_triplets(data):
    """
    Validate Triplets (Anchor = Positive != Negative)
    """
    for d in data:
        a_id, p_id, n_id = d["anchor"]["id"], d["positive"]["id"], d["negative"]["id"]
        a_path, p_path, n_path = d["anchor"]["path"], d["positive"]["path"], d["negative"]["path"]
        if a_id != p_id or a_id == n_id:
            raise Exception(f"Invalid Entry: {d}")

        if a_path == p_path or a_path == n_path or p_path == n_path:
            [print(f"{p[0]} - {p[1]}") for p in [("a_path", a_path), ("p_path", p_path), ("n_path", n_path)]]
            raise Exception(f"Invalid Entry (same Path): {d}")


if __name__ == "__main__":
    def list_mango_entries_by_cat(category_name):
        entries = walk_entries(rf"{MANGO_BASE_PATH}\{category_name}")
        anchor_items = (list(filtered_entries(entries)))
        return anchor_items


    def list_asos_entries_by_cat(category_name):
        def clean_entry(entry):
            return {"id": entry["url"].replace(Asos_Selectors.URLS.BASE, ""),
                    "images": [{"path": brand_path.relative_image_path_from_url(img["url"]), "view": img["description"]}
                               for img in entry["images"]]}

        brand_path = AsosPaths(ASOS_BASE_PATH)
        entries_db_path = brand_path.get_entries_db_base_path()
        entries = list_dbs_by_category(entries_db_path, ASOS_CATEGORIES)
        entries_by_cat_distinct = distinct_list_of_dicts(flatten([Json_DB(x).all() for x in entries[category_name]]),
                                                         key="url")

        anchor_items = list(map(clean_entry, entries_by_cat_distinct))
        return anchor_items


    def list_hm_entries_by_cat(category_name):
        def clean_entry(entry):
            return {"id": entry["url"].replace(HM_Selectors.URLS.BASE, ""),
                    "images": [{"path": brand_path.relative_image_path_from_url(img["url"]), "view": img["name"]}
                               for img in entry["images"]]}

        brand_path = HMPaths(HM_BASE_PATH)
        entries_db_path = brand_path.get_entries_db_base_path()
        entries = list_dbs_by_category(entries_db_path, HM_CATEGORIES)
        entries_by_cat_distinct = distinct_list_of_dicts(flatten([Json_DB(x).all() for x in entries[category_name]]),
                                                         key="url")

        anchor_items = list(map(clean_entry, entries_by_cat_distinct))
        return anchor_items


    anchor_items = list_asos_entries_by_cat("hose")
    triplets_df = as_df(build_triplets(anchor_items))

    print(triplets_df.head(3))

    print("Columns", triplets_df.columns)
    print("Len", len(triplets_df))
