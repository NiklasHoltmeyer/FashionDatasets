from multiprocessing import RLock
from pathlib import Path

from fashiondatasets.utils.io import list_dir_abs_path
from fashionscrapper.utils.io import json_load
from tqdm.auto import tqdm

tqdm.set_lock(RLock())


class Annotations:
    @staticmethod
    def list(annotations_path):
        """

        :return: List of Annotations::Path
        """
        return list_dir_abs_path(annotations_path)

    @staticmethod
    def list_images_from_annotations(images_path, annotations, IGNORE_CHECK):
        """

        :param images_path: Base Path for all Images
        :param annotations: List of Annotations
        :param IGNORE_CHECK: sanity check, if images exists
        :return: List of Image::Path (JPG)
        """
        annotations_len = len(annotations)
        annotations_f_names = map(lambda x: x.name, annotations)
        imgs = map(lambda x: x.split(".json")[0] + ".jpg", annotations_f_names)
        imgs = list(map(lambda x: Path(images_path, x), imgs))

        if not IGNORE_CHECK:
            imgs_itter = tqdm(imgs, desc="IMG::exists", total=annotations_len)

            img_missing = any(filter(lambda i: not i.exists(), imgs_itter))
            assert not img_missing, "At least one Image missing"

        return imgs

    @staticmethod
    def list_with_images(annotations_path, images_path, IGNORE_CHECK):
        annotations = Annotations.list(annotations_path)
        imgs = Annotations.list_images_from_annotations(images_path=images_path,
                                                        annotations=annotations,
                                                        IGNORE_CHECK=IGNORE_CHECK)
        return list(zip(annotations, imgs))

    @staticmethod
    def load(path, flatten_item_n=False):
        annotation_data = json_load(path)
        if not flatten_item_n:
            return annotation_data
        return Annotations.flatten_items(annotation_data)

    @staticmethod
    def flatten_items(annotation):
        data = {"items": []}

        for k, v in annotation.items():
            if "item" in k:
                data["items"].append(v)
            else:
                data[k] = v

        return data

    @staticmethod
    def drop_keys(annotation, key):

        if key == "segmentation":
            item_keys = filter(lambda i: "item" in i, annotation.keys())
            list(
                map(lambda item_key: annotation[item_key].pop("segmentation", None), item_keys)
            )

    @staticmethod
    def drop_list_of_keys(annotation, keys):
        return list(
            map(lambda k: Annotations.drop_keys(annotation, k), keys)
        )
