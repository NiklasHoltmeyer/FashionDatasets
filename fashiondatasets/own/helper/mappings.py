import tensorflow as tf


def preprocess_image(img_shape, preprocess_img=None, channels=3, dtype=tf.float32):
    @tf.function
    def __call__(filename):
        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=channels)
        image = tf.image.convert_image_dtype(image, dtype)
        image = tf.image.resize(image, img_shape)
        if preprocess_img:
            image = preprocess_img(image)
        return image

    return __call__
