import os
from collections import defaultdict
from pathlib import Path
import json
from fashionscrapper.utils.list import distinct

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
        grouped_entries = self._group_entries(entries)
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
    def _group_entries(entries):
        item_ids_by_split = defaultdict(lambda: [])
        split_by_id = defaultdict(lambda: [])
        images_by_id = defaultdict(lambda: defaultdict(lambda: []))

        for entry in entries:
            image_1, image_2, _id, split = entry
            item_ids_by_split[split].append(_id)
            split_by_id[_id].append(split)

            images = [image_1, image_2]
            for image_group in [CONSUMER, "shop"]:
                imgs_by_group = list(filter(lambda i: image_group in i, images))
                images_by_id[_id][image_group].extend(imgs_by_group)

        images_by_id_distinct = {}

        for article_id, img_dict in images_by_id.items():
            clean_img_dict = {
                k: distinct(v) for k, v in img_dict.items()
            }

            images_by_id_distinct[article_id] = clean_img_dict

        grouped_data = defaultdict(lambda: [])
        for _id, c2s in images_by_id_distinct.items():
            splits = distinct(split_by_id[_id])
            assert len(splits) == 1
            split = splits[0]
            grouped_data[split].append({_id: c2s})

        flatten_split = lambda split_data: {k: v for d in split_data for k, v in d.items()}

        return {k: flatten_split(v) for k, v in grouped_data.items()}

    def category_mappings(self):
        img_base_path = Path(self.base_path, "Img", "img")

        cat_name_by_idxs = defaultdict(lambda: {})
        cat_idx_by_name = defaultdict(lambda: {})

        for tl_idx, tl_cat in enumerate(os.listdir(img_base_path)):
            cat_name = tl_cat  # "/".join([tl_cat, cat])
            cat_name_by_idxs[tl_idx] = cat_name
            cat_idx_by_name[cat_name] = tl_idx
#            for idx, cat in enumerate(os.listdir(img_base_path / tl_cat)):
#                # cat_name_by_idxs[tl_idx][idx] = cat_name
#                # cat_idx_by_name[cat_name] = (tl_idx, idx)

        return cat_name_by_idxs, cat_idx_by_name

    def encode_labels(self, grouped_entries):
        cat_name_by_idxs, cat_idx_by_name = self.category_mappings()

        for split, split_data in grouped_entries.items():
            for pair_id, images in split_data.items():
                sample_img = images["shop"][0]
                # cat_groups = sample_img.split("/")[-4:-2]
                cat_groups = sample_img.split("/")[1]

                # images["cat_name"] = "/".join(cat_groups)
                images["cat_name"] = cat_groups
                images["cat_idx"] = cat_idx_by_name[images["cat_name"]]

        return grouped_entries, cat_name_by_idxs, cat_idx_by_name

    @staticmethod
    def generate_ids_by_cat(grouped_entries):
        # ids_by_cat_split = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))
        ids_by_cat_split = defaultdict(lambda: defaultdict(lambda: []))

        for split, split_data in grouped_entries.items():
            for _id, data in split_data.items():
                cat_idx = data["cat_idx"]
                ids_by_cat_split[split][cat_idx].append(_id)
                # ids_by_cat_split[split][cat_idx[0]][cat_idx[1]].append(_id)

        return ids_by_cat_split


if __name__ == "__main__":
    base_path = r"D:\Download\Cts"
    print(DF1_Split_Extractor(base_path).load_helper(force=True))
