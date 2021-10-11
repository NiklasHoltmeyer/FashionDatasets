# TODO just create this from scratch. hacked this down asap

import copy
from pathlib import Path
from random import shuffle

import numpy as np
# from fashionscrapper.brand import parser_settings
from fashionscrapper.utils.io import Json_DB
from fashionscrapper.utils.list import idx_self_reference
from tqdm.auto import tqdm
from collections import defaultdict
from random import choice

from fashiondatasets.utils.list import random_range, flatten_dict
import pandas as pd


class EntriesHelper:
    def __init__(self, entries):
        self.entries_original = entries
        self.entries = copy.deepcopy(entries)

        self.random_all_ids = None
        self.random_all_ids_lst = None
        self.entries_by_category = None
        self.entries_by_id = None
        self.number_of_images_by_id = None
        self.all_ids = None
        self.id_imgs_mapping = None

        self.load_data()

    def _load_random_ids(self):
        self.random_all_ids = copy.deepcopy(self.all_ids)
        shuffle(self.random_all_ids)

        self.random_all_ids_lst = [self.random_all_ids, copy.deepcopy(self.random_all_ids)]

        self.random_id_hits = len(self.random_all_ids)

    def _load_entries_by_cat(self):
        _entries_by_id = copy.deepcopy(self.entries_by_id)

        self.entries_by_category = defaultdict(lambda: [])
        for id, entry in _entries_by_id.items():
            category = entry["category"]
            self.entries_by_category[category].append(id)

        [shuffle(v) for k, v in self.entries_by_category.items()]

        self.entries_by_category_lst = [self.entries_by_category,
                                        copy.deepcopy(self.entries_by_category)]

    def load_data(self):
        self.entries_by_id = {e["id"]: {**e} for e in self.entries}
        self.number_of_images_by_id = {e["id"]: len(e["images"]) for e in self.entries}
        self.all_ids = [e for e in self.entries_by_id]

        assert len(self.entries_by_id) == len(entries) == len(self.all_ids)
        self.id_imgs_mapping = {id: random_range(self.number_of_images_by_id[id]) for id in self.all_ids}

        self._load_random_ids()
        self._load_entries_by_cat()

    def random_category(self, blacklist=None):
        keys = list(self.entries_by_category.keys())

        max_retries = 100

        while True:
            assert (max_retries := (max_retries - 1)) > 1, "MAX RETRIES random_category"
            key = choice(keys)
            if not blacklist or key not in blacklist:
                return key

    def get_random_all_ids_lst(self, negative_idx):
        return self.random_all_ids_lst[negative_idx]

    def get_entries_by_category_lst(self, negative_idx, cat_name):
        return self.entries_by_category_lst[negative_idx][cat_name]


def _random_references(num_idxs, max_retries=150):
    random_idxs = list(range(num_idxs))
    for _ in range(max_retries):
        shuffle(random_idxs)  # in place
        if not idx_self_reference(random_idxs):
            break
    return np.array(random_idxs)  # copy -> None otherwise


def filter_not_none(lst):  # delete
    return filter(lambda x: x is not None, lst)


def build_quadruplets(entries_helper):
    def positive_image_pairs(entry_id):
        n_images = entries_helper.number_of_images_by_id[entry_id]
        anchors = range(n_images)
        positive_anchors = _random_references(n_images)
        pairs = []
        for pair in zip(anchors, positive_anchors):
            if (pair[1], pair[0]) not in pairs:
                pairs.append(pair)

        return entry_id, [{"anchor": {"id": entry_id, "img": a}, "positive": {"id": entry_id, "img": p}} for (a, p) in
                          pairs]

    def load_negative_id(anchor_id, category, same_category):
        negative_id = None
        target_category = category if same_category else entries_helper.random_category(category)
        max_retries = 100

        while not negative_id:
            assert (max_retries := (max_retries - 1)) > 1, "Max Retries"

            possible_item = entries_helper.entries_by_category[target_category].pop(0)
            if possible_item != anchor_id:
                negative_id = possible_item
            entries_helper.entries_by_category[target_category].append(possible_item)

        img = entries_helper.id_imgs_mapping[negative_id].pop(0)
        entries_helper.id_imgs_mapping[negative_id].append(img)

        return {'id': negative_id, 'img': img}

    def negative_image_pairs(anchor_id, positive_pair):
        anchor_cat = entries_helper.entries_by_id[anchor_id]["category"]

        negative = load_negative_id(anchor_id, anchor_cat, same_category=True)
        negative2 = load_negative_id(anchor_id, anchor_cat, same_category=False)

        negatives = list(filter_not_none([negative, negative2]))

        return {**positive_pair, "negatives": negatives}

    def validate_quadruplet(apnn):
        assert len(apnn.keys()) == 3  # Anchor, Positive, Negatives
        a_id, p_id = apnn["anchor"]["id"], apnn["positive"]["id"]

        a_cat = entries_helper.entries_by_id[a_id]["category"]

        assert a_id == p_id, "Anchor and Positive must be Same Article!"

        negative_ids = list(map(lambda d: d["id"], apnn["negatives"]))
        n_ids_neq_a_id = all(map(lambda id: a_id != id, negative_ids))

        assert n_ids_neq_a_id, "Negative_ID == A_ID"

        negative_categories = map(lambda id: entries_helper.entries_by_id[id], negative_ids)
        negative_categories = map(lambda entry: entry["category"], negative_categories)

        n_dif_cat = len(list(filter(lambda cat: cat != a_cat, negative_categories)))
        n_same_dif = len(apnn["negatives"]) - n_dif_cat

        if n_dif_cat < 1 or n_same_dif < 1:
            return False
        return True

    errors, succ = 0, 0

    for anchor_id in tqdm(entries_helper.all_ids):
        anchor_id, positive_pairs = positive_image_pairs(anchor_id)

        for pair in positive_pairs:
            apnn = negative_image_pairs(anchor_id, pair)
            if validate_quadruplet(apnn):
                succ += 1
                yield apnn
            else:
                errors += 1

    print("Errors:", errors)
    print("Success:", succ, f"{(100 * succ) / (errors + succ)}%")


def unzip_quadruplets_nb(entries_helper, quadruplet, base_path=""):
    # {'anchor': {'id': 50371, 'img': 0}, 'positive': {'id': 50371, 'img': 3}, 'negatives': [{'id': 87380, 'img': 0}, {'id': 97536, 'img': 2}]}
    def load_img(d):
        header, item = d

        id, img_id = item["id"], item["img"]
        entry = entries_helper.entries_by_id[id]
        cat = entry["category"]
        row = {"id": id, **entry["images"][img_id], "category": entry["category"]}
        row = {f"{header}_{k}": v for (k, v) in row.items()}
        row[f"{header}_path"] = row[f"{header}_path"].replace(base_path, "")
        # dicts[0]["a_path"].replace(BP, "")
        return row

    a, p, ns = quadruplet["anchor"], quadruplet["positive"], quadruplet["negatives"]
    header = ["a", "p", "n1", "n2"]
    items = [a, p, *ns]
    items_img = list(map(load_img, zip(header, items)))
    return flatten_dict(items_img)


def to_csv(base_path, quadruplets):
    df_path = Path(base_path, "quadruplet.csv")
    dicts = map(lambda x: unzip_quadruplets_nb(entries_helper, x, str(base_path)), quadruplets)
    dicts = list(dicts)
    quadruplets_df = pd.DataFrame(dicts, columns=dicts[0].keys())

    quadruplets_df.to_csv(df_path, sep=";", index=False)
    print("Saved Quadruplets to", df_path)


if __name__ == "__main__":
    BP = "F:\\workspace\\datasets\\own"

    entries_path = Path(BP, "entires.json")
    with Json_DB(entries_path) as entries_db:
        entries = entries_db.all()
        #^ list of dicts. required keys: id, images, category
        # images = [{"path": ...}]
        print(entries[0].keys())
        exit(0)
    shuffle(entries)
    entries_helper = EntriesHelper(entries)

    quadruplets = build_quadruplets(entries_helper)
    to_csv(BP, quadruplets)
