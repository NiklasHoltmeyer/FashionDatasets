import argparse

import os
from multiprocessing.dummy import freeze_support
from pathlib import Path

import albumentations as A
import numpy as np
from fashionscrapper.utils.list import flatten
from fashionscrapper.utils.parallel_programming import calc_chunk_size
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from fashiondatasets.utils.io import load_img, save_image, list_dir_abs_path
from fashiondatasets.utils.list import parallel_map


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

    parser.add_argument(
        '--sub_folders',
        dest='sub_folders',
        help='Subfolders (e.g. Images, Annotations, ...)',
        nargs='+',
        default=["annotations", "images"])

    parser.add_argument(
        '--validate_images',
        dest='validate_images',
        help='Check wether all Images can be loaded (PIL)',
        type=bool,
        default=False)

    parser.add_argument(
        '--validate_images_force',
        dest='validate_images_force',
        help='Check wether all Images can be loaded (PIL). [Force = Check all Files]',
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


def validate_images(imgs, threads):
    def validate_image(img):
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

#    chunk_size = calc_chunk_size(n_workers=threads, len_iterable=len(imgs))

#    r = thread_map(validate_image, imgs, max_workers=threads, total=len(imgs),
#                   chunksize=chunk_size, desc=f"Transform Images ({threads} Threads)")

    n_successful = sum(r)
    print(f"{n_successful} / {len(imgs)} = {100 * n_successful / len(imgs)}%  Validated")
    return n_successful == len(imgs)

def validate_all_images(args, threads=os.cpu_count()):
    def all_images(args):
        dst = Path(args.dst_path)

        for folder in args.sub_folders:
            folder_path = dst / folder
            for img in os.listdir(folder_path):
                yield folder_path / img

    return validate_images(all_images(args), threads)


def batch_transform(args, transform, threads=os.cpu_count()):
    src, dst = Path(args.src_path), Path(args.dst_path)
    #    logger = defaultLogger("Batch Transform Images")
    assert src.exists()

    assert all(map(lambda x: (src / x).exists(), args.sub_folders)), "At least one Folder doesnt exist"

    def resize_jobs(folders):
        for folder in folders:
            is_mask = "anno" in folder or "label" in "folder"
            (dst / folder).mkdir(exist_ok=True, parents=True)

            for file in os.listdir(src / folder):
                yield src / folder / file, dst / folder / file, is_mask

    def filter_not_dst_exists(job):
        return not job[1].exists()

    #    logger.debug("List Images")
    print("List Images")
    jobs = list(resize_jobs(args.sub_folders))
    jobs = filter(filter_not_dst_exists, tqdm(jobs, desc="Filter DST::Exists", total=len(jobs)))
    jobs = list(jobs)

    hide_exceptions = False  # len(jobs) > 100

    #    logger.debug("Transform Images")
    print("Transform Images")

    if len(jobs) < 1:
        return True

    r = parallel_map(
        lst=jobs,
        fn=transform_image(transform, hide_exceptions),
        desc="Transform Images",
        threads=threads
    )
#    chunk_size = calc_chunk_size(n_workers=threads, len_iterable=len(jobs))
#    r = thread_map(transform_image(transform, hide_exceptions), jobs, max_workers=threads, total=len(jobs),
#                   chunksize=chunk_size, desc=f"Transform Images ({threads} Threads)")

    n_successful = sum(r)
    #    logger.debug(f"{n_succ} / {len(jobs)} = {100*n_succ/len(jobs)}%  Resized")

    print(f"{n_successful} / {len(jobs)} = {100 * n_successful / len(jobs)}%  Resized")

    if args.validate_images:
        target_imgs = map(lambda j: j[1], jobs)
        return validate_images(target_imgs, threads)
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
        print("[Force] Validate all DST Images")
        validate_all_images(args, threads=os.cpu_count())

#Train (for Mask and Quad):
# --src_path "F:\workspace\datasets\DeepFashion2 Dataset\train"
# --dst_path "F:\workspace\datasets\DeepFashion2 Dataset\train_256"
# --sub_folders "annotations" "images" --validate_images True
# --validate_images_force True

#Val (for Quad):
# --src_path "F:\workspace\datasets\DeepFashion2 Dataset\validation"
# --dst_path "F:\workspace\datasets\DeepFashion2 Dataset\validation_256"
# --sub_folders "images" --validate_images True
# --validate_images_force True

