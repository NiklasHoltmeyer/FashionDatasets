import json
import os
from collections import defaultdict
from pathlib import Path

CONSUMER = "comsumer"  # <- they mis spelled it. so to make sure everyting works just use this const


class DF1_Split_Extractor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.consumer2shop_items_path = Path(base_path, "Anno", "list_item_consumer2shop.txt")
        self.list_eval_partition = Path(base_path, "Eval", "list_eval_partition.txt")

    def load_helper(self, force=False):
        splits_file = Path(self.base_path, "splits.json")  # ids_by_cat_idx
        if not splits_file.exists() or force:
            self.dump_split()

        dicts = ["splits", "cat_name_by_idxs", "cat_idx_by_name", "ids_by_cat_idx"]
        data = {}

        for dict in dicts:
            with open(Path(self.base_path, dict + ".json"), ) as f:
                data[dict] = json.load(f)

        return data

    def dump_split(self):
        entries = self._retrieve_entries()
        grouped_entries = self.group_by_split(entries)
        grouped_entries, cat_name_by_idxs, cat_idx_by_name = self.encode_labels(grouped_entries)
        ids_by_cat_idx = self.generate_ids_by_cat(grouped_entries)

        with open(Path(self.base_path, "splits.json"), "w") as f:
            json.dump(grouped_entries, f)

        with open(Path(self.base_path, "cat_name_by_idxs.json"), "w") as f:
            json.dump(cat_name_by_idxs, f)

        with open(Path(self.base_path, "cat_idx_by_name.json"), "w") as f:
            json.dump(cat_idx_by_name, f)

        with open(Path(self.base_path, "ids_by_cat_idx.json"), "w") as f:
            json.dump(ids_by_cat_idx, f)

    def _retrieve_entries(self):
        with open(self.list_eval_partition, "r") as f:
            lines = f.readlines()
            lines = list(map(lambda l: l.replace("\n", ""), lines))

        unpack_lines = lambda l: [i for i in l.split(" ") if len(i) > 0]
        # n_images = lines[0]
        # header = lines[1]
        entries = lines[2:]
        entries = list(map(unpack_lines, entries))

        assert all([len(e) == 4 for e in entries])

        return entries

    @staticmethod
    def group_by_split(entries):
        r = defaultdict(lambda: [])
        for i1, i2, _id, split in entries:
            assert (CONSUMER in i1)
            assert ("shop" in i2)
            assert _id in i1
            assert _id in i2
            r[split].append((i1, i2, _id))
        return dict(r)

    def category_mappings(self):
        img_base_path = Path(self.base_path, "Img", "img")

        cat_name_by_idxs = defaultdict(lambda: {})
        cat_idx_by_name = defaultdict(lambda: {})

        for tl_idx, tl_cat in enumerate(os.listdir(img_base_path)):
            cat_name = tl_cat  # "/".join([tl_cat, cat])
            cat_name_by_idxs[tl_idx] = cat_name
            cat_idx_by_name[cat_name] = tl_idx

        return cat_name_by_idxs, cat_idx_by_name

    def encode_labels(self, grouped_entries):
        cat_name_by_idxs, cat_idx_by_name = self.category_mappings()

        r = {}
        for split, split_data in grouped_entries.items():
            r[split] = []
            for c_img, s_img, pair_id in split_data:
                cat_name = c_img.split("/")[1]
                cat_idx = cat_idx_by_name[cat_name]
                r[split].append({
                    "anchor": c_img,
                    "positive": s_img,
                    "cat_idx": cat_idx,
                    "cat_name": cat_name,
                    "pair_id": pair_id
                })
        return r, cat_name_by_idxs, cat_idx_by_name

    @staticmethod
    def generate_ids_by_cat(grouped_entries):
        ids_by_cat_split = defaultdict(lambda: defaultdict(lambda: []))

        for split, split_data in grouped_entries.items():
            for data in split_data:
                ids_by_cat_split[split][data["cat_idx"]].append(data)

        return ids_by_cat_split


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_1_256"
    print(DF1_Split_Extractor(base_path).load_helper(force=True))
