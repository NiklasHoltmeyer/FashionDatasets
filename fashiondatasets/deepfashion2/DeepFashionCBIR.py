from pathlib import Path

from fashionscrapper.utils.io import json_load
from fashionscrapper.utils.list import flatten
import tensorflow as tf
from fashiondatasets.utils.list import parallel_map


class DeepFashionCBIR:
    def __init__(self, base_path, split_suffix=""):
        self.base_path = base_path

        self.test_path = Path(base_path, "test" + split_suffix)
        self.images_path = self.test_path / Path("images")

        self.retrieval_information = self.load_retrieval_information(self.test_path)
        self.split_suffix = split_suffix

    def load_retrieval_information(self, test_path):
        retrieval_information_names = {
            "queries": Path(test_path, "retrieval_test_consumer_information.json"),
            "gallery": Path(test_path, "retrieval_test_shop_information.json")
        }

        assert all(filter(lambda p: p.exists(), retrieval_information_names.values()))

        def clean_image(data):  # ["file_name", "id"]
            return {
                "id": data["id"],
                "path": str((self.images_path / data["file_name"]).resolve())
            }

        clean_images = lambda lst: list(map(clean_image, lst))
        load_clean_images = lambda p: clean_images(json_load(p)["images"])

        retrieval_information = {k: load_clean_images(v) for k, v in retrieval_information_names.items()}
        return retrieval_information

    def load_as_dataset(self):
        assert len(self.retrieval_information.keys()) == 2
        # unpack = lambda d: (d["id"], d["path"])
        get_id = lambda d: d["id"]
        get_path = lambda d: d["path"]

        queries, gallery = self.retrieval_information["queries"], self.retrieval_information["gallery"]

        dataset = {
            "query": {
                "ids": (list(map(get_id, queries))),
                "paths": tf.data.Dataset.from_tensor_slices(list(map(get_path, queries)))
            },
            "gallery": {
                "ids": (list(map(get_id, gallery))),
                "paths": tf.data.Dataset.from_tensor_slices(list(map(get_path, gallery)))
            }
        }

        return dataset

    def load(self, validate_paths=False):
        if validate_paths:
            self.validate_image_paths()

        return self.retrieval_information

    def validate_image_paths(self):
        path_exists = lambda p: Path(p).exists()

        data = flatten(self.retrieval_information.values())
        image_paths = list(map(lambda d: d["path"], data))

        r = parallel_map(lst=image_paths, fn=path_exists, desc="Validating Image-Paths")

        if all(r):
            return True

        ratio = 100 * sum(r) / len(image_paths)
        print(f"{sum(r)} / {len(image_paths)} Images Exist. ({ratio:.2f}%)")

        return False


if __name__ == "__main__":
    base_path = r"F:\workspace\datasets\deep_fashion_256"
    results = DeepFashionCBIR(base_path, split_suffix="_256").load(False)
    flatten()