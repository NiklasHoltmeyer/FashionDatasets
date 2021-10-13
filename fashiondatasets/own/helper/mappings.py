import tensorflow as tf

def preprocess_image(img_shape, channels=3, dtype=tf.float32):
    def __call__(filename):
        """
        Load the specified file as a JPEG image, preprocess it and
        resize it to the target shape.
        """

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image_string, channels=channels)
        image = tf.image.convert_image_dtype(image, dtype)
        image = tf.image.resize(image, img_shape)
        return image
    return __call__