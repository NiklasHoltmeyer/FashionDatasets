import os
from multiprocessing.dummy import freeze_support
from pathlib import Path
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


def parallel_map(lst, fn, desc=None, total=None, threads=None, disable_output=False):
    desc = "Parallel-Map" if desc is None else desc
    threads = threads if threads else os.cpu_count()

    total = total if total else len(lst)
    chunk_size = calc_chunk_size(n_workers=threads, len_iterable=total)

    r = thread_map(fn, lst, max_workers=threads, total=total,
                   chunksize=chunk_size, desc=f"{desc} ({threads} Threads)",
                   disable=disable_output)

    return r


def filter_not_exist(lst, parallel=True, not_exist=True, key=lambda i: i, **p_map_kwargs):
    def check_map(i):
        value = key(i)
        exist = Path(value).exists()
        filter = exist ^ not_exist
        # exist and not_exist -> False
        # exist and !not_exist -> True

        # !exist and not_exist -> True
        # !exist and !not_exist -> False

        if filter:
            return i
        return None

    if parallel:
        r = parallel_map(lst=lst, fn=check_map, **p_map_kwargs)
    else:
        r = map(check_map, lst)

    return list(filter(lambda x: x is not None, r))


if __name__ == "__main__":
    x = ['F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_01.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_02.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_03.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_04.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_05.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_06.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_07.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_08.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_09.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00000218\\comsumer_10.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00003453\\comsumer_10.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00006499\\comsumer_03.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00007562\\shop_01.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00010145\\shop_01.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00010236\\shop_01.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00011436\\comsumer_03.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00012494\\comsumer_01.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00014247\\shop_02.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00015525\\shop_02.jpg',
         'F:\\workspace\\datasets\\deep_fashion_1_256\\img_256\\img\\TOPS\\T_Shirt\\id_00020957\\shop_01.jpg']

    filter_not_exist(x, False, key=lambda d: d)
