from tensorflow.keras.preprocessing.image import load_img
from IPython.display import Image, display
import PIL
from PIL import ImageOps
from PIL import Image as PILImage
from random import randint
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

def mask_to_img(mask, is_prediction, num_classes, resize=None):
    """
    SRC: Keras
    Quick utility to display a model's prediction."""
    if is_prediction:
        mask = np.argmax(mask, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
    msk = PIL.ImageOps.autocontrast(array_to_img(mask))

    if num_classes:
        color_map = get_color_map_list(num_classes)
        msk.putpalette(color_map)

    if resize:
        msk = msk.resize((144, 144))

    return msk.convert("RGB")

def display_mask_prediction(prediction, num_classes=None):
    mask = mask_to_img(prediction, num_classes, is_prediction=True)
    display(mask)

def blend_mask(img, msk, num_classes=None, alpha=0.5):
    msk = PIL.ImageOps.autocontrast(msk)

    if num_classes:
        color_map = get_color_map_list(num_classes)
        msk.putpalette(color_map)

    msk = msk.convert("RGB")

    return PILImage.blend(img, msk, alpha=alpha)

def get_color_map_list(num_classes):
    """ Src: PaddleSeq """
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


