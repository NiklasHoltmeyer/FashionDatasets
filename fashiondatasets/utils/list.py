from random import shuffle


def random_range(n):
    range_lst = list(range(n))
    shuffle(range_lst)
    return range_lst

def flatten_dict(dict_):
    return {k: v for d in dict_ for k, v in d.items()}
