import tensorflow as tf
import numpy as np

def build_pairs_ds_fn(is_triplet, is_ctl):
    """
    Params: is_triplet: Triplet_loss, else Quad.
    :return: Zipped Dataframe Consisting of A, P, N or A, P, N1, N2 depending on is_triplet Flag
    """

    def zip_triplets(a, p, n, ctls=None):
        a_ds = tf.data.Dataset.from_tensor_slices(a)

        if not ctls:
            p_ds = tf.data.Dataset.from_tensor_slices(p)
            n_ds = tf.data.Dataset.from_tensor_slices(n)

            return tf.data.Dataset.zip((a_ds, p_ds, n_ds))

        a_ctl, p_ctl, n_ctl = ctls
        #a_ctl_ds = tf.data.Dataset.from_tensor_slices(a_ctl)
        p_ctl_ds = tf.data.Dataset.from_tensor_slices(p_ctl)
        n_ctl_ds = tf.data.Dataset.from_tensor_slices(n_ctl)

        #return tf.data.Dataset.zip((a_ds, p_ds, n_ds, a_ctl_ds, p_ctl_ds, n_ctl_ds))
        return tf.data.Dataset.zip((a_ds, p_ctl_ds, n_ctl_ds))

    def zip_quadruplets(a, p, n1, n2, ctls=None):
        a_ds = tf.data.Dataset.from_tensor_slices(a)
        n1_ds = tf.data.Dataset.from_tensor_slices(n1)

        if not ctls:
            p_ds = tf.data.Dataset.from_tensor_slices(p)
            n2_ds = tf.data.Dataset.from_tensor_slices(n2)

            return tf.data.Dataset.zip((a_ds, p_ds, n1_ds, n2_ds))

        a_ctl, p_ctl, n1_ctl, n2_ctl = ctls
        #a_ctl_ds = tf.data.Dataset.from_tensor_slices(a_ctl)
        p_ctl_ds = tf.data.Dataset.from_tensor_slices(p_ctl)
        n1_ctl_ds = tf.data.Dataset.from_tensor_slices(n1_ctl)
        n2_ctl_ds = tf.data.Dataset.from_tensor_slices(n2_ctl)

        return tf.data.Dataset.zip((a_ds, n1_ds, p_ctl_ds, n1_ctl_ds, n2_ctl_ds))

    def apnn_pairs(a, p, n1, n2, ctls=None):
        if is_ctl and not ctls:
            raise Exception("No CTLS provided")

        return zip_quadruplets(a, p, n1, n2, ctls=ctls)

    def apn_pairs(a, p, n1, n2, ctls=None):
        if is_ctl and not ctls:
            raise Exception("No CTLS provided")

        n = []
        if ctls is None:
            for i, (n1, n2) in enumerate(zip(n1, n2)):
                if i % 2 == 0:
                    n.append(n1)
                else:
                    n.append(n2)
            return zip_triplets(a, p, n, ctls=None)
        else:
            a_ctl, p_ctl, n1_ctl, n2_ctl = ctls
            new_ctls = []
            for i, (n1, n2, n1_ctl_, n2_ctl_) in enumerate(zip(n1, n2, n1_ctl, n2_ctl)):
                if i % 2 == 0:
                    n.append(n1)
                    new_ctls.append(n1_ctl_)
                else:
                    n.append(n2)
                    new_ctls.append(n2_ctl_)
            ctls_ = [a_ctl, p_ctl, new_ctls]
            return zip_triplets(a, p, n, ctls=ctls_)

    if is_triplet:
        return apn_pairs
    return apnn_pairs
