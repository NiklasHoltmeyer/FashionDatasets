from multiprocessing.dummy import freeze_support
from pathlib import Path

from fashiondatasets.deepfashion2.deepfashion2_preprocessor import DeepFashion2Preprocessor

if __name__ == "__main__":
    freeze_support()

    annotations_path, images_path = (Path(f'F:/workspace/datasets/DeepFashion2 Dataset/train/annos'),
                                     Path(f'F:/workspace/datasets/DeepFashion2 Dataset/train/image'))

    coco_train_path = r"F:\workspace\datasets\DeepFashion2 Dataset\train\train_coco.json"

    preprocessor_settings = {
        "annotations_path": annotations_path,
        "images_path": images_path,
        "IGNORE_CHECK": True,
        "threads": 8,
    }

    preprocessor = DeepFashion2Preprocessor(**preprocessor_settings)
    preprocessor.semantic_segmentation(coco_train_path)
