import math

from fashiondatasets.deepfashion2.helper.pairs.deep_fashion_pairs_generator import DeepFashionPairsGenerator


class DeepFashionIterator:
    def __init__(self, _base_path, embedding, split, batch_size, is_triplet, split_suffix="", **kwargs):
        self.generator = DeepFashionPairsGenerator(_base_path, embedding, split_suffix="", **kwargs)
        self.batch_size = batch_size
        self.split = split
        self.is_triplet = is_triplet

    def __len__(self):
        _len = len(self.generator.df_helper[self.split].user.image_ids)
        return math.ceil(_len / self.batch_size)

    def __iter__(self):
        """
        Forces the Creation of Hard-Triplets after Each Epoch!
        :return:
        """
        if self.is_triplet:
            data = self.generator.build_anchor_positive_negative(self.split)
        else:
            data = self.generator.build_anchor_positive_negative1_negative2(self.split)

        for d in data:
            yield d



