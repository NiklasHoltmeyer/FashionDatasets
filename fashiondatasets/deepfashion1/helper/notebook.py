import os
import pickle
from pathlib import Path

from fashiondatasets.deepfashion1.DeepFashion1CBIR import DeepFashion1CBIR
from tqdm.auto import tqdm

def map_top_k(result, k=20):
    matches = result["distances"]["matches"]
    return min([x[-1] for x in matches]) < k


def calc_top_k_from_distances(results):
    _result = results["result"]
    top_20 = map_top_k

    top_100 = lambda r: map_top_k(r, k=100)
    top_50 = lambda r: map_top_k(r, k=50)

    top_20_value = 100 * sum(map(top_20, _result)) / len(_result)
    top_50_value = 100 * sum(map(top_50, _result)) / len(_result)
    top_100_value = 100 * sum(map(top_100, _result)) / len(_result)

    top_values = {
        "embedding_path": results["embedding_path"],
        "top_20": top_20_value,
        "top_50": top_50_value,
        "top_100": top_100_value
    }

    return top_values

def distances_existence_filtered(path, exist=False):
    x = list(map(lambda p: os.path.join(path, p), os.listdir(path)))
    x = list(filter(lambda p: os.path.isdir(p), x))
    x = list(map(lambda p: (p, os.path.join(p, "distances.pkl")), x))

    if exist:
        existence_filter = lambda p: Path(p[1]).exists()
    else:
        existence_filter = lambda p: not Path(p[1]).exists()

    x = filter(existence_filter, x)
    x = list(x)

    return x

def calc_top_k_from_embeddings(base_path, embedding_path, idx=None, pickle_result=True, raise_exception=True):
    idx = idx or 0

    print(embedding_path)
    distances_path = os.path.join(embedding_path, "distances.pkl")

    cbir = DeepFashion1CBIR(base_path,
                            model=None,
                            embedding_path=embedding_path,
                            disable_output=idx > 0)
    result = {
        "embedding_path": embedding_path
    }

    try:
        distance_walker = cbir.walk_distances()
        result["result"] = list(distance_walker)
    except Exception as e:
        if raise_exception:
            raise e
        result["exception"] = e

    if not pickle_result:
        return result

    try:
        with open(distances_path, "wb") as f:
            pickle.dump(result, f)
    except:
        print("Pickle, failed")

    return result

def load_distances(embedding_path):
    distance_paths = [x[1] for x in distances_existence_filtered(embedding_path, exist=True)]

    for d_p in tqdm(distance_paths, desc="Load Distances"):
        with open(d_p, "rb") as f:
            data = pickle.load(f)
            yield data

def load_results(embedding_path, verbose=False):
    results = []

    for distance in load_distances(embedding_path):
        top_values = calc_top_k_from_distances(distance)

        if verbose:
            top_values.update(distance)

        results.append(top_values)

    return results

if __name__ == "__main__":
    emb_base = r"C:\workspace"
    print(load_results(emb_base))


