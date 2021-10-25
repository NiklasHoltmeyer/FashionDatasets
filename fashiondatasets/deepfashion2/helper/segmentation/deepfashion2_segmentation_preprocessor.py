from multiprocessing import RLock

import albumentations as A
from fashionscrapper.utils.parallel_programming import calc_chunk_size
from pycocotools.coco import COCO
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from fashiondatasets.deepfashion2.helper.annotations import Annotations
from fashiondatasets.utils.io import load_img, save_image
from fashiondatasets.utils.list import parallel_map
from fashiondatasets.utils.logger.defaultLogger import defaultLogger

tqdm.set_lock(RLock())


class DeepFashion2SegmentationPreprocessor:
    def __init__(self, **settings):
        self.annotations_path = settings.pop("annotations_path")
        self.images_path = settings.pop("images_path")
        self.threads = settings.pop("threads", 2)
        self.settings = settings
        self.logger = defaultLogger("DeepFashion2 Preprocessor")

    @staticmethod
    def BboxParams(min_area=1024, min_visibility=0.1):
        return A.BboxParams(format='albumentations', label_fields=['class_labels', 'class_categories'],  # coco
                            min_area=min_area, min_visibility=min_visibility, )

    def bounding_boxes(self, transform):
        annotation_imgs = Annotations.list_with_images(self.annotations_path, self.images_path,
                                                       IGNORE_CHECK=self.settings.get("IGNORE_CHECK", False))
        r = parallel_map(lst=annotation_imgs,
                         fn=transform_w_bb(transform),
                         desc="Transform Bounding-Boxes")

        r_true = filter(lambda x: x[0], r)
        r_true = map(lambda x: x[1:], r_true)  # removing first list_item per list (successful-Flag)
        r_true = list(r_true)
        r_false = filter(lambda x: not x[0], r)
        r_false = map(lambda x: x[1:], r_false)
        r_false = list(r_false)

        n_transformed = len(r_true)
        total = n_transformed + len(r_false)

        self.logger.debug(f"Transformed-Images: {n_transformed} / {total} = {100 * n_transformed / total}%")

        return r_true, r_false

    def semantic_segmentation(self, coco_path):
        img_mask_dir = self.images_path.parent / "annotations"
        img_mask_dir.mkdir(parents=True, exist_ok=True)

        coco = COCO(coco_path)

        coco_images = list(coco.imgs.values())
        mask_does_not_exist = lambda img: not (img_mask_dir / img["file_name"].replace(".jpg", ".png")).exists()
        coco_images = filter(mask_does_not_exist, tqdm(coco_images, desc="Filter Mask::exists", total=len(coco_images)))
        coco_images = list(coco_images)
        len_iterable = len(coco_images)

        self.logger.debug(f"{len_iterable} Annotations")  #

        chunk_size = calc_chunk_size(n_workers=self.threads, len_iterable=len_iterable)

        r = thread_map(save_segmentation_mask(coco, img_mask_dir), coco_images, max_workers=self.threads,
                       chunksize=chunk_size, desc=f"Transform Segmentation ({self.threads} Threads)")

        n_mask_created = sum(r)
        self.logger.debug(f"{n_mask_created} Mask Created. {len_iterable - n_mask_created} Failed.")


def clean_bbox(w, h):
    # coco: [x_min, y_min, width, height]#albumentations/voc [x_min, y_min, x_max, y_max] <- (0, 1)
    def __call__(bbox):
        x_min, y_min, x_max, y_max = bbox
        x_min, y_min, x_max, y_max = x_min / w, y_min / h, x_max / w, y_max / h
        x_min, y_min = max(x_min, 0.0), max(y_min, 0.0)
        x_max, y_max = min(x_max, 1.0), min(y_max, 1.0)

        return [x_min, y_min, x_max, y_max]

    return __call__


def clean_bboxes(bboxes, width, height):
    clean_bbox_ = clean_bbox(width, height)
    return list(map(clean_bbox_, bboxes))


def load_items(annotation_path):
    annotation_data = Annotations.load(annotation_path)
    item_keys = filter(lambda i: "item" in i, annotation_data.keys())
    items = map(lambda k: annotation_data[k], item_keys)

    return list(items)


def transform_image_bounding_box(annotation_path, image_path, transformer):
    img = (load_img(image_path))
    height, width, channels = img.shape

    annotation_items = load_items(annotation_path)

    bboxes = map(lambda i: i["bounding_box"], annotation_items)
    bboxes = clean_bboxes(bboxes, width, height)

    category_ids = map(lambda i: i["category_id"], annotation_items)
    category_ids = list(category_ids)

    category_names = map(lambda i: i["category_name"], annotation_items)
    category_names = list(category_names)

    try:
        return True, transformer(image=img, bboxes=bboxes, class_labels=category_ids,
                                 class_categories=category_names)
    except Exception as e:
        return False, (annotation_path, image_path, str(e))


def transform_w_bb(transformer):
    def __call__(d):
        return transform_image_bounding_box(d[0], d[1], transformer)

    return __call__


def save_mask(img_mask_dir, img, mask):
    f_name = img["file_name"].replace(".jpg", ".png")
    mask_file_path = str((img_mask_dir / f_name).resolve())
    #    save_image_PMODE(mask, mask_file_path)
    save_image(mask, mask_file_path)


def save_segmentation_mask(coco, img_mask_dir, ignore_exceptions=True):
    cat_ids = coco.getCatIds()

    # noinspection SpellCheckingInspection
    def __call__(img):
        try:
            anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=False)
            anns = coco.loadAnns(anns_ids)

            mask = coco.annToMask(anns[0])
            for i in range(len(anns)):
                mask += coco.annToMask(anns[i])

            save_mask(img_mask_dir, img, mask)
            return 1
        except Exception as e:
            if ignore_exceptions:
                return 0
            raise e

    return __call__
