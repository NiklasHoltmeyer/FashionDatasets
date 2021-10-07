from tensorflow.keras.preprocessing.image import load_img
from IPython.display import Image, display
import PIL
from PIL import ImageOps
from random import randint
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img


def display_mask_prediction(prediction):
    """
    SRC: Keras
    Quick utility to display a model's prediction."""
    mask = np.argmax(prediction, axis=-1)
    print(mask.shape)
    mask = np.expand_dims(mask, axis=-1)
    print(mask.shape)
    img = PIL.ImageOps.autocontrast(array_to_img(mask))
    display(img)
