import matplotlib.pyplot as plt
import numpy as np
from . import plot_config as config

clr = config.colour_dict
c = config.default_colour


def cluster_sizes(sizes, percentages, out_path):
    fig, ax = plt.subplots(1,1, figsize=(len(sizes)*1.6, 6))

    # Clusters numbered 0-N, so make appropriate axis.
    x_axis = np.arange(len(sizes))
    # Plot cluster sizes with percentages as labels
    bar = ax.bar(x_axis, sizes, color=c)
    plt.bar_label(bar, labels=[f"{x:.1f}%" for x in percentages])

    ax.set_xlabel('Cluster No.')
    ax.set_xticks(x_axis)
    ax.set_ylabel('Cluster Sizes')
    ax.set_ylim(0, sizes[0]+.2*sizes[0])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()

