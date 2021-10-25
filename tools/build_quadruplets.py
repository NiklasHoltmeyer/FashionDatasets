# TODO just create this from scratch. hacked this down asap

import copy
import os
from collections import defaultdict
from multiprocessing.dummy import freeze_support
from pathlib import Path
from random import choice
from random import shuffle

import albumentations as A
import numpy as np
import pandas as pd
from fashionscrapper.utils.io import Json_DB
from fashionscrapper.utils.list import flatten
from fashionscrapper.utils.list import idx_self_reference, distinct
from fashionscrapper.utils.parallel_programming import calc_chunk_size
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from fashiondatasets.own.Quadruplets import Quadruplets
from fashiondatasets.utils.io import load_img, save_image
from fashiondatasets.utils.list import random_range, flatten_dict


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
        for _id, entry in _entries_by_id.items():
            category = entry["category"]
            self.entries_by_category[category].append(_id)

        [shuffle(v) for k, v in self.entries_by_category.items()]

        self.entries_by_category_lst = [self.entries_by_category,
                                        copy.deepcopy(self.entries_by_category)]

    def load_data(self):
        self.entries_by_id = {e["id"]: {**e} for e in self.entries}
        self.number_of_images_by_id = {e["id"]: len(e["images"]) for e in self.entries}
        self.all_ids = [e for e in self.entries_by_id]

        assert len(self.entries_by_id) == len(self.entries) == len(self.all_ids)
        self.id_imgs_mapping = {_id: random_range(self.number_of_images_by_id[_id]) for _id in self.all_ids}

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
        ap_pairs_distinct = []
        for ap_pairs in zip(anchors, positive_anchors):
            if (ap_pairs[1], ap_pairs[0]) not in ap_pairs:
                ap_pairs_distinct.append(ap_pairs)

        return entry_id, [{"anchor": {"id": entry_id, "img": a}, "positive": {"id": entry_id, "img": p}} for (a, p) in
                          ap_pairs_distinct]

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

    def negative_image_pairs(_anchor_id, positive_pair):
        anchor_cat = entries_helper.entries_by_id[_anchor_id]["category"]

        negative = load_negative_id(_anchor_id, anchor_cat, same_category=True)
        negative2 = load_negative_id(_anchor_id, anchor_cat, same_category=False)

        negatives = list(filter_not_none([negative, negative2]))

        return {**positive_pair, "negatives": negatives}

    def validate_quadruplet(_apnn):
        assert len(_apnn.keys()) == 3  # Anchor, Positive, Negatives
        a_id, p_id = _apnn["anchor"]["id"], _apnn["positive"]["id"]

        a_cat = entries_helper.entries_by_id[a_id]["category"]

        assert a_id == p_id, "Anchor and Positive must be Same Article!"

        negative_ids = list(map(lambda d: d["id"], _apnn["negatives"]))
        n_ids_neq_a_id = all(map(lambda _id: a_id != _id, negative_ids))

        assert n_ids_neq_a_id, "Negative_ID == A_ID"

        negative_categories = map(lambda _id: entries_helper.entries_by_id[_id], negative_ids)
        negative_categories = map(lambda entry: entry["category"], negative_categories)

        n_dif_cat = len(list(filter(lambda cat: cat != a_cat, negative_categories)))
        n_same_dif = len(_apnn["negatives"]) - n_dif_cat

        if n_dif_cat < 1 or n_same_dif < 1:
            return False
        return True

    errors, n_successful = 0, 0

    for anchor_id in tqdm(entries_helper.all_ids):
        anchor_id, positive_pairs = positive_image_pairs(anchor_id)

        for pair in positive_pairs:
            apnn = negative_image_pairs(anchor_id, pair)
            if validate_quadruplet(apnn):
                n_successful += 1
                yield apnn
            else:
                errors += 1

    print("Errors:", errors)
    print("Success:", n_successful, f"{(100 * n_successful) / (errors + n_successful)}%")


def unzip_quadruplets_nb(entries_helper, quadruplet, base_path=""):
    # {'anchor': {'_id': 50371, 'img': 0}, 'positive': {'_id': 50371, 'img': 3}, 'negatives': [{'_id': 87380, 'img': 0},
    # {'_id': 97536, 'img': 2}]}

    def load_image(d):
        header, item = d

        _id, img_id = item["id"], item["img"]
        entry = entries_helper.entries_by_id[_id]
        row = {"id": _id, **entry["images"][img_id], "category": entry["category"]}
        row = {f"{header}_{k}": v for (k, v) in row.items()}
        row[f"{header}_path"] = row[f"{header}_path"].replace(base_path, "")
        # dicts[0]["a_path"].replace(BP, "")
        return row

    a, p, ns = quadruplet["anchor"], quadruplet["positive"], quadruplet["negatives"]
    header = ["a", "p", "n1", "n2"]
    items = [a, p, *ns]
    items_img = list(map(load_image, zip(header, items)))
    return flatten_dict(items_img)


def to_csv(entries_helper, base_path, quadruplets):
    df_path = Path(base_path, "quadruplet_full.csv")
    dicts = map(lambda x: unzip_quadruplets_nb(entries_helper, x, str(base_path)), quadruplets)
    dicts = list(dicts)
    quadruplets_df = pd.DataFrame(dicts, columns=dicts[0].keys())

    quadruplets_df.to_csv(df_path, sep=";", index=False)
    print("Saved Quadruplets to", df_path)


def transform_quads(base_path, target_path, transformer, validate=True):
    Path(target_path).mkdir(parents=True, exist_ok=True)

    def _list_jobs(path):
        quad = Quadruplets(path)
        p_keys = Quadruplets.list_path_column_keys(quad.df)
        relative_paths = map(lambda k: quad.df[k].values, p_keys)
        relative_paths = distinct(flatten(relative_paths))

        build_job = lambda p: (base_path + p, target_path + p)

        return list(map(build_job, relative_paths))

    def transform_image(_transformer, hide_exceptions):
        def __call__(_job):
            src, dst = _job
            try:
                img = np.array(load_img(src))
                img_transformed = _transformer(image=img)["image"]

                save_image(img_transformed, dst, create_parents=True)

                return 1
            except Exception as e:
                if hide_exceptions:
                    return 0
                raise e

        return __call__

    _transformer = transform_image(transformer, True)
    jobs = _list_jobs(base_path)

    # noinspection PyBroadException
    def map_exists_validate(_job):
        dst_exists = Path(_job[1]).exists()
        if not dst_exists:
            return _job

        if validate:
            try:
                load_img(_job[1])
                return None
            except:
                Path(_job[1]).unlink()
                return _job

    def add_file_ext(_job):
        ext = os.path.splitext(_job[1])[-1]
        if len(ext) < 1:
            return _job[0], _job[1] + ".jpg"
        return _job[0], _job[1]

    threads = os.cpu_count() / 2
    jobs, total = map(add_file_ext, jobs), len(jobs)

    chunk_size = calc_chunk_size(n_workers=threads, len_iterable=total)
    jobs_validated = thread_map(map_exists_validate, jobs, max_workers=threads, total=total,
                                chunksize=chunk_size, desc=f"Validate-Paths Images ({threads} Threads)")
    jobs_validated = list(filter_not_none(jobs_validated))
    chunk_size = calc_chunk_size(n_workers=threads, len_iterable=len(jobs_validated))
    jobs_transformed = list(thread_map(_transformer, jobs_validated, max_workers=threads, total=len(jobs_validated),
                                       chunksize=chunk_size, desc=f"Transform Images ({threads} Threads)"))

    n_successful = sum(jobs_transformed)

    if len(jobs_validated) < 10:
        for job in jobs_validated:
            print("DST", job[1])

    print(f"{n_successful} / {len(jobs_validated)} = {100 * n_successful / len(jobs_validated)}%  Transformed")


if __name__ == "__main__":
    BP, target = "F:\\workspace\\datasets\\own", r"F:\workspace\datasets\own_256"


    def build_quads():
        entries_path = Path(BP, "entries.json")
        with Json_DB(entries_path) as entries_db:
            entries = entries_db.all()
            # ^ list of dicts. required keys: id, images, category
            # images = [{"path": ...}]
        shuffle(entries)
        entries_helper = EntriesHelper(entries)

        quadruplets = build_quadruplets(entries_helper)
        to_csv(entries_helper, BP, quadruplets)


    freeze_support()

    transform = A.Compose([
        A.Resize(width=256, height=256),
        # A.RandomCrop(width=244, height=244),
    ])

    transform_quads(BP, target, transform)
