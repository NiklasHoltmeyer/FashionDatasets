from scipy.spatial import distance as distance_metric


def find_top_k(queries, gallery, reverse, k=20):
    """ Reverse=True -> return Top-K most Sim. Reverse=False -> Top-k most dissimilar"""
    distances = distance_metric.cdist(queries, gallery, "cosine")
    most_similar_idxs = []
    for distance in distances:
        distance = 1 - distance
        idx_dist = list(zip(range(len(gallery)), distance))
        idx_dist = sorted(idx_dist, key=lambda d: d[1], reverse=reverse)[:k]
        most_sim_idxs = list(map(lambda d: d[0], idx_dist))
        most_similar_idxs.append(most_sim_idxs)
    return most_sim_idxs
