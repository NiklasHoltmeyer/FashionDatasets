import argparse

import os
from multiprocessing.dummy import freeze_support
from pathlib import Path

import albumentations as A
import numpy as np
from fashionscrapper.utils.parallel_programming import calc_chunk_size
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from datasets.deepfashion2.deepfashion2_preprocessor import save_image_PMODE
from datasets.utils.io import load_img, save_image

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

    return parser.parse_args()


def filter_dst_not_exists(job):
    return not job[1].exists()


def transform_image(transformer, hide_exceptions):
    def __call__(job):
        src, dst, is_mask = job
        try:
            img = np.array(load_img(src))
            img_transformed = transformer(image=img)["image"]

            if is_mask:
                save_image_PMODE(img_transformed, str(dst))
            else:
                save_image(img_transformed, dst)

            return 1
        except Exception as e:
            if hide_exceptions:
                return 0
            raise e

    return __call__


def batch_transform(args, transform, threads=os.cpu_count() * 2):
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

    chunk_size = calc_chunk_size(n_workers=threads, len_iterable=len(jobs))

    #    logger.debug("Transform Images")
    print("Transform Images")

    if len(jobs) < 1:
        exit(0)

    r = thread_map(transform_image(transform, hide_exceptions), jobs, max_workers=threads, total=len(jobs),
                   chunksize=chunk_size, desc=f"Transform Images ({threads} Threads)")

    n_successful = sum(r)
    #    logger.debug(f"{n_succ} / {len(jobs)} = {100*n_succ/len(jobs)}%  Resized")

    print(f"{n_successful} / {len(jobs)} = {100 * n_successful / len(jobs)}%  Resized")


if __name__ == "__main__":
    args = parse_args()

    freeze_support()
    transform = A.Compose([
        A.Resize(width=256, height=256),
        # A.RandomCrop(width=244, height=244),
    ])

    batch_transform(args, transform)


