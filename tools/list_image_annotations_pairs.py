import os
from pathlib import Path
from random import shuffle


def list_image_annotations_pairs(ds_path, image_dir_name, label_dir_name):
    image_file_names = os.listdir(Path(ds_path, image_dir_name))
    label_file_names = os.listdir(Path(ds_path, label_dir_name))

    assert len(image_file_names) == len(label_file_names), "Len(Images) != Len(Labels)"

    def same_file_name(img_lbl, IGNORE_FILEFORMAT=True):
        img, lbl = img_lbl
        if IGNORE_FILEFORMAT:
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


if __name__ == "__main__":
    dataset_path = r"F:\workspace\datasets\DeepFashion2 Dataset\train"

    split = {
        "train": 0.8,
        "val": 0.2,
        "test": 0.0
    }

    image_dir_name = "images"
    label_dir_name = "annotations"

    sep = " "

    img_annotation_pairs = list_image_annotations_pairs(dataset_path, image_dir_name, label_dir_name)
    img_annotation_pairs = list(map(lambda x: sep.join(x) + "\n", img_annotation_pairs))

    splitted_data = split_pairs(img_annotation_pairs, split)

    for split, pairs in splitted_data.items():
        with open(Path(dataset_path, split+".txt"), 'w+') as f:
            f.writelines(pairs)

        with open(Path(dataset_path, split+".txt"), 'r') as f:
            assert (len(list(f.readlines()))) == len(pairs)
