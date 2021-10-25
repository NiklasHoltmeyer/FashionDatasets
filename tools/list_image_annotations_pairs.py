import argparse
import os
from pathlib import Path
from random import shuffle


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Export Image Annotations Pairs as TXT'
    )

    parser.add_argument(
        '--ds_path',
        dest='dataset_path',
        help='Base Dataset Path',
        type=str,
        required=True)

    parser.add_argument(
        '--split',
        dest='split',
        help='Desired Split [Train, Validate, Test] e.g. default 0.7, 0.15, 0.15',
        nargs=3,
        type=float,
        required=True,
        default=[0.7, 0.15, 0.15]
    )

    parser.add_argument(
        '--image_dir_name',
        dest='image_dir_name',
        help='Name of Image (Input) Folder.',
        type=str,
        required=False,
        default="images"
    )

    parser.add_argument(
        '--label_dir_name',
        dest='label_dir_name',
        help='Name of Image (Input) Folder.',
        type=str,
        required=False,
        default="annotations"
    )

    parser.add_argument(
        '--sep',
        dest='sep',
        help='Separator',
        type=str,
        required=False,
        default=" "
    )

    return parser.parse_args()


def list_image_annotations_pairs(ds_path, image_dir_name, label_dir_name):
    image_file_names = os.listdir(Path(ds_path, image_dir_name))
    label_file_names = os.listdir(Path(ds_path, label_dir_name))

    assert len(image_file_names) == len(label_file_names), "Len(Images) != Len(Labels)"

    def same_file_name(img_lbl, IGNORE_FILE_FORMAT=True):
        img, lbl = img_lbl
        if IGNORE_FILE_FORMAT:
            return img.split(".")[0] == lbl.split(".")[0]
        return img == lbl

    image_labels = list(zip(image_file_names, label_file_names))
    assert all(map(same_file_name, image_labels)), "Annotations != Imgs"

    def relative_paths(img_lbl):
        img, lbl = img_lbl
        return f"{image_dir_name}/{img}", f"{label_dir_name}/{lbl}"

    image_labels = map(relative_paths, image_labels)

    return list(image_labels)


def split_pairs(pairs, splits, shuffle_pairs=True):
    assert sum(splits.values()) == 1.0

    if shuffle_pairs:
        shuffle(pairs)

    train_samples = int(splits["train"] * len(pairs))
    validate_samples = int(splits["val"] * len(pairs))
    test_samples = int(splits["test"] * len(pairs))

    train_samples += (len(pairs) - train_samples - validate_samples - test_samples)

    ds = {
        "train": pairs[:train_samples],
        "val": pairs[train_samples:-validate_samples],
        "test": pairs[-validate_samples:]
    }

    assert (len(ds["train"]) + len(ds["val"]) + len(ds["test"])) == len(pairs)

    return ds


def save_pairings_to_txt(_args):
    split = {
        "train": _args.split[0],
        "val": _args.split[1],
        "test": _args.split[2]
    }

    img_annotation_pairs = list_image_annotations_pairs(_args.dataset_path, _args.image_dir_name, _args.label_dir_name)
    img_annotation_pairs = list(map(lambda x: _args.sep.join(x) + "\n", img_annotation_pairs))

    splitted_data = split_pairs(img_annotation_pairs, split)

    for split, pairs in splitted_data.items():
        with open(Path(_args.dataset_path, split + ".txt"), 'w+') as f:
            f.writelines(pairs)

        with open(Path(_args.dataset_path, split + ".txt"), 'r') as f:
            assert (len(list(f.readlines()))) == len(pairs)


if __name__ == "__main__":
    args = parse_args()
    save_pairings_to_txt(args)
