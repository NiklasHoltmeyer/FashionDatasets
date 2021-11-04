import tensorflow as tf


def _build_pairs_ds_fn(is_triplet):
    """
    Params: is_triplet: Triplet_loss, else Quad.
    :return: Zipped Dataframe Consisting of A, P, N or A, P, N1, N2 depending on is_triplet Flag
    """

    def zip_triplets(a, p, n):
        a_ds = tf.data.Dataset.from_tensor_slices(a)
        p_ds = tf.data.Dataset.from_tensor_slices(p)
        n_ds = tf.data.Dataset.from_tensor_slices(n)

        return tf.data.Dataset.zip((a_ds, p_ds, n_ds))

    def zip_quadruplets(a, p, n1, n2):
        a_ds = tf.data.Dataset.from_tensor_slices(a)
        p_ds = tf.data.Dataset.from_tensor_slices(p)
        n1_ds = tf.data.Dataset.from_tensor_slices(n1)
        n2_ds = tf.data.Dataset.from_tensor_slices(n2)

        return tf.data.Dataset.zip((a_ds, p_ds, n1_ds, n2_ds))

    def apnn_pairs(a, p, n1, n2):
        return zip_quadruplets(a, p, n1, n2)

    def apn_pairs(a, p, n1, n2):
        n = []
        for i, (n1, n2) in enumerate(zip(n1, n2)):
            if i % 2 == 0:
                n.append(n1)
            else:
                n.append(n2)
        return zip_triplets(a, p, n)

    if is_triplet:
        return apn_pairs
    return apnn_pairs
