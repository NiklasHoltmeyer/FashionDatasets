from collections import defaultdict
from pathlib import Path

import numpy as np
from fashiondatasets.deepfashion2.helper.pairs.similar_embeddings import calculate_most_similar
from fashionscrapper.utils.list import flatten, distinct

from fashiondatasets.utils.list import parallel_map

def group_images_by_ids(list_split_data, image_key):
    images_by_id = defaultdict(lambda: [])
    for split_data in list_split_data:
        for d in split_data:
            img, pair_id = [d[k] for k in [image_key, 'pair_id']]
            images_by_id[pair_id].append(img)
    return dict(images_by_id)


def build_gallery(splits):
    # splits = [splits["val"], splits["test"]]
    images_by_id = group_images_by_ids(splits, "positive")

    all_shop_images = lambda lst: all(map(lambda i: "shop" in i, lst))
    assert all(map(all_shop_images, images_by_id.values()))

    return images_by_id


def build_queries(splits):
    images_by_id = group_images_by_ids(splits, "anchor")

    no_shop_images = lambda lst: all(map(lambda i: "shop" not in i, lst))
    assert all(map(no_shop_images, images_by_id.values()))

    return images_by_id

def flatten_distinct_values(dictionary):
    return distinct(flatten(dictionary.values()))

#def remove_duplicate_embeddings(embeddings):
#    len_embeddings = len(embeddings)
#    d = defaultdict(lambda: [])
#    for p, e in embeddings:
#        d[p].append(e)

#    d = {k: v for k, v in d.items()}

#    identity_fn = lambda x: x

#    d_distinct = {}

#    for k, v in d.items():
#        first_embedding = v.pop()
#        if len(v) > 1: # just check if all embeddings are the same, if true remove duplicates
#            distances = calculate_most_similar(
                #first_embedding,
                #v,
                #embedding_key=identity_fn,
                #idx_key=identity_fn,
                #k=len_embeddings,
                #most_similar=True
#            )

#            assert (sum([x[1] for x in distances[1]])) == 0.0

#        d_distinct[k] = first_embedding
#    return d_distinct


def jpg_to_npy_path(embedding_path, img_path):
    clean_f_name = img_path.replace("img/", "").replace("/", "-").replace(".jpg", ".npy")
    return str(Path(embedding_path, clean_f_name).resolve())

#def npy_to_jpg_path(embedding_path, npy_path):
#    if not type(embedding_path) == str:
#        embedding_path = str(embedding_path.resolve())

#    jpg_path = "/img/" + npy_path.replace(embedding_path, "").replace("-", "/").replace(".npy", ".jpg")
#    return jpg_path

def save_batch_encodings(batch_encodings, embedding_path):
    def save_job(d):
        emb_path, embedding = d
        np.save(emb_path, embedding)

    save_jobs = []
    for path, embedding in batch_encodings.items():
        emb_path = jpg_to_npy_path(embedding_path, path)
        save_jobs.append((emb_path, embedding))

    parallel_map(lst=save_jobs, fn=save_job, desc="Saving Encoding Batch")

def flatten_gallery(gallery_items):
    data = [[(pair_id, i) for i in images] for pair_id, images in gallery_items]
    data = flatten(data)

    pair_ids, images = list(zip(*data))
    assert len(pair_ids) == len(images)
    return pair_ids, images

def load_embedding_from_image_path(embedding_path, image_path):
    npy_path = jpg_to_npy_path(embedding_path, image_path)
    return np.load(npy_path)




