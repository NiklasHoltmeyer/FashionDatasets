import argparse
import os
from multiprocessing.dummy import freeze_support
from pathlib import Path

import albumentations as A
import numpy as np
from tqdm.auto import tqdm

from fashiondatasets.utils.io import load_img, save_image
from fashiondatasets.utils.list import parallel_map
from fashiondatasets.utils.logger.defaultLogger import defaultLogger

logger = defaultLogger("fashion_image_preprocess")

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Batch Transform Images'
    )

    parser.add_argument(
        '-i',
        '--src_path',
        dest='src_path',
        help='Base Source Path',
        type=str,
        required=True)

    parser.add_argument(
        '-o',
        '--dst_path',
        dest='dst_path',
        help='Base Destination Path',
        type=str,
        required=True)

    #    parser.add_argument(
    #        '--sub_folders',
    #        dest='sub_folders',
    #        help='Sub-Folders (e.g. Images, Annotations, ...)',
    #        nargs='+',
    #        default=["annotations", "images"])

    parser.add_argument(
        '--validate_images',
        dest='validate_images',
        help='Check whether all Images can be loaded (PIL)',
        type=bool,
        default=False)

    parser.add_argument(
        '--validate_images_force',
        dest='validate_images_force',
        help='Check whether all Images can be loaded (PIL). [Force = Check all Files]',
        type=bool,
        default=False)

    parser.add_argument(
        '--validate_images_retries',
        dest='validate_images_retries',
        help='Number of Retries. (Batch-Transform + Validate)*N',
        type=int,
        default=3)

    return parser.parse_args()


def filter_dst_not_exists(job):
    return not job[1].exists()


def transform_image(transformer, hide_exceptions):
    def __call__(job):
        src, dst, is_mask = job
        try:
            img = np.array(load_img(src))
            img_transformed = transformer(image=img)["image"]

            save_image(img_transformed, dst)

            return 1
        except Exception as e:
            if hide_exceptions:
                return 0
            raise e

    return __call__


def validate_images(imgs):
    def validate_image(img):
        # noinspection PyBroadException
        try:
            img = load_img(img)
            return 1
        except:
            img.unlink()
            return 0

    imgs = list(imgs)

    r = parallel_map(lst=imgs,
                     fn=validate_image,
                     desc="Transform Images")

    n_successful = sum(r)
    logger.info(f"{n_successful} / {len(imgs)} = {100 * n_successful / len(imgs)}%  Validated")
    return n_successful == len(imgs)


def validate_all_images(_args):
    def all_images(__args):
        dst = __args.dst_path

        for i_src, i_dst, is_mask in walk_jobs(dst, dst):
            yield i_dst

    return validate_images(all_images(_args))


def walk_jobs(src_base, dst_base):
    src_base = str(Path(src_base).resolve())
    dst_base = str(Path(dst_base).resolve())

    for root, directory, files in os.walk(src_base):
        parent_created = False

        # is_mask = "anno" in folder or "label" in "folder"

        for file in files:
            if ".jpg" in file:
                folder_name = Path(root).name
                is_mask = "anno" in folder_name or "label" in folder_name

                img_src = Path(root, file)
                img_dst = str(img_src.resolve()).replace(src_base, dst_base)
                img_dst = Path(img_dst)

                if not parent_created:
                    img_dst.parent.mkdir(exist_ok=True, parents=True)
                    parent_created = True

                yield img_src, img_dst, is_mask


def batch_transform(_args, _transform, threads=os.cpu_count()):
    src, dst = Path(_args.src_path), Path(_args.dst_path)
    #    logger = defaultLogger("Batch Transform Images")
    assert src.exists()

    #    assert all(map(lambda x: (src / x).exists(), _args.sub_folders)), "At least one Folder doesnt exist"

    #    def resize_jobs(folders): # doesnt work for nisted folders
    #        for folder in folders:
    #            # noinspection SpellCheckingInspection
    #            is_mask = "anno" in folder or "label" in "folder"
    #            (dst / folder).mkdir(exist_ok=True, parents=True)

    #            for file in os.listdir(src / folder):
    #                yield src / folder / file, dst / folder / file, is_mask

    def filter_not_dst_exists(job):
        return not job[1].exists()

    #    logger.debug("List Images")
    logger.info("List Images")
    #    jobs = list(resize_jobs(_args.sub_folders))
    jobs = list(walk_jobs(src, dst))
    jobs = filter(filter_not_dst_exists, tqdm(jobs, desc="Filter DST::Exists", total=len(jobs)))
    jobs = list(jobs)

    hide_exceptions = False  # len(jobs) > 100

    #    logger.debug("Transform Images")
    logger.info("Transform Images")

    if len(jobs) < 1:
        return True

    r = parallel_map(
        lst=jobs,
        fn=transform_image(_transform, hide_exceptions),
        desc="Transform Images",
        threads=threads
    )

    n_successful = sum(r)

    logger.info(f"{n_successful} / {len(jobs)} = {100 * n_successful / len(jobs)}%  Resized")

    if _args.validate_images:
        target_imgs = map(lambda j: j[1], jobs)
        return validate_images(target_imgs)
    else:
        return n_successful == len(jobs)


if __name__ == "__main__":
    args = parse_args()

    freeze_support()

    transform = A.Compose([
        A.Resize(width=256, height=256),
        # A.RandomCrop(width=244, height=244),
    ])

    for _ in range(args.validate_images_retries):
        if batch_transform(args, transform):
            break

    if args.validate_images_force:
        logger.info("[Force] Validate all DST Images")
        validate_all_images(args)

# Train (for Mask and Quad):
# --src_path "F:\workspace\datasets\deep_fashion_256\train"
# --dst_path "F:\workspace\datasets\deep_fashion_256\train_256"
# --sub_folders "annotations" "images" --validate_images True
# --validate_images_force True

# Val (for Quad):
# --src_path "F:\workspace\datasets\deep_fashion_256\validation"
# --dst_path "F:\workspace\datasets\deep_fashion_256\validation_256"
# --sub_folders "images" --validate_images True
# --validate_images_force True

# Test (for Quad):
# --src_path "F:\workspace\datasets\deep_fashion_256\test"
# --dst_path "F:\workspace\datasets\deep_fashion_256\test_256"
# --sub_folders "images" --validate_images True
# --validate_images_force True
