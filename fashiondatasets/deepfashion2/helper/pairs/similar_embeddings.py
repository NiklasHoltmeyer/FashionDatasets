from scipy.spatial import distance as distance_metric



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
