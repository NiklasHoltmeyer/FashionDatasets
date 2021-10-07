import PIL
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from PIL import ImageOps

def display_from_path(file_path, autocontrast=False):
    image = Image(filename=str(file_path)) if not autocontrast \
                else PIL.ImageOps.autocontrast(load_img(str(file_path)))
    display(image)