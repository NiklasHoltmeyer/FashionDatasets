import matplotlib.pyplot as plt


def visualize(anchors, positives, *negative_lst):
    # Adapted for Quad. from: https://keras.io/examples/vision/siamese_network/
    def show(ax, image):
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    n_negative_samples = len(negative_lst)

    cols = (2 + n_negative_samples)
    rows = min(len(anchors), cols)

    fig = plt.figure(figsize=(9, 9))
    axs = fig.subplots(rows, cols)

    for i in range(rows):
        show(axs[i, 0], anchors[i])
        show(axs[i, 1], positives[i])
        for n_idx, negative in enumerate(negative_lst):
            show(axs[i, n_idx + 2], negative[i])
