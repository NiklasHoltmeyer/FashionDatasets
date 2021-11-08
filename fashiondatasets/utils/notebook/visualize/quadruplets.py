from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def visualize(anchors, positive, negatives1, negatives2=None, fig_size=(9, 9), base_path=None):
    n_samples = len(anchors)
    if type(anchors[0]) == str:
        if base_path:
            full_path = lambda p: str(Path(base_path, p).resolve())
        else:
            full_path = lambda p: p
        load_image = lambda p: mpimg.imread(full_path(p))
    else:
        load_image = lambda i: i

    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=fig_size)
    if negatives2:
        axs = fig.subplots(n_samples, 4)
    else:
        axs = fig.subplots(n_samples, 3)

    for i in range(n_samples):
        show(axs[i, 0], load_image(anchors[i]))
        show(axs[i, 1], load_image(positive[i]))
        show(axs[i, 2], load_image(negatives1[i]))
        if negatives2:
            show(axs[i, 3], load_image(negatives2[i]))


def unzip_pairs(pairs):
    return list(zip(*pairs))
