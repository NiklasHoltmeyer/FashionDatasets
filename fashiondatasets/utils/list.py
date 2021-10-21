import os
from multiprocessing.dummy import freeze_support
from random import shuffle

from fashionscrapper.utils.parallel_programming import calc_chunk_size
from tqdm.contrib.concurrent import thread_map

freeze_support()


def random_range(n):
    range_lst = list(range(n))
    shuffle(range_lst)
    return range_lst


def flatten_dict(dict_):
    return {k: v for d in dict_ for k, v in d.items()}


def parallel_map(lst, fn, desc="Parallel-Map", total=None, threads=None):
    threads = threads if threads else os.cpu_count()

    total = total if total else len(lst)
    chunk_size = calc_chunk_size(n_workers=threads, len_iterable=total)

    r = thread_map(fn, lst, max_workers=threads, total=total,
                   chunksize=chunk_size, desc=f"{desc} ({threads} Threads)")

    return r
