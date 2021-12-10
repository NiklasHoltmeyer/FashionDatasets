import copy
from collections import defaultdict
from random import choice
from random import shuffle

from fashiondatasets.utils.list import random_range


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
            max_retries -= 1
            assert max_retries > 1, "MAX RETRIES random_category" # walrus operator does not work on colab

            key = choice(keys)
            if not blacklist or key not in blacklist:
                return key

    def get_random_all_ids_lst(self, negative_idx):
        return self.random_all_ids_lst[negative_idx]

    def get_entries_by_category_lst(self, negative_idx, cat_name):
        return self.entries_by_category_lst[negative_idx][cat_name]
