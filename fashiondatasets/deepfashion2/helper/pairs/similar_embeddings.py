from scipy.spatial import distance as distance_metric
import numpy as np

from fashiondatasets.utils.list import parallel_map


def find_top_k(queries, gallery, most_similar, k=20):
    reverse = not most_similar
    distances = distance_metric.cdist(queries, gallery, "sqeuclidean")
    list_of_idxs = []

    for distance in distances:
        idx_dist = list(zip(range(len(gallery)), distance))
        idx_dist = sorted(idx_dist, key=lambda d: d[1], reverse=reverse)[:k]
        most_sim_idxs = list(map(lambda d: d[0], idx_dist))
        list_of_idxs.append(most_sim_idxs)

    return list_of_idxs


def calculate_most_similar(query, gallery,
                           embedding_key=None,
                           idx_key=None,
                           k=101,
                           most_similar=True,
                           debugging=False):
    if not embedding_key:
        embedding_key = lambda d: d

    if not idx_key:
        idx_key = lambda d: d

    query_gallery_distances = []

    q_emb = embedding_key(query)

    for gallery_data in gallery:
        g_emb = embedding_key(gallery_data)

        dist = np.linalg.norm(q_emb - g_emb)  # euklidische distanz
        idx = idx_key(gallery_data)

        query_gallery_distances.append((idx, dist))

    query_gallery_distances = sorted(query_gallery_distances, key=lambda d: d[1], reverse=not most_similar)
    query_gallery_distances = query_gallery_distances[:k]

    return idx_key(query), query_gallery_distances


if __name__ == "__main__":
    query = [
        ("1", np.array([0, 0, 0]))
    ]

    gallery = [
        (str(id_), np.array([id_] * 3)) for id_ in range(10)
    ]

    query, retrieved_results = (calculate_most_similar(query[0], gallery, embedding_key=lambda d: d[1], idx_key=lambda d: d[0],))
    for result in retrieved_results:
        print(result)
