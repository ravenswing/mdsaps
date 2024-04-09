import matplotlib.pyplot as plt
import numpy as np
from . import plot_config as config

colours = config.nspc


def cluster_sizes(sizes, percentages, out_path):
    fig, ax = plt.subplots(1,1, figsize=(len(sizes)*1.6, 6))

    # Clusters numbered 0-N, so make appropriate axis.
    x_axis = np.arange(len(sizes))
    # Plot cluster sizes with percentages as labels
    bar = ax.bar(x_axis, sizes, color=colours.default)
    plt.bar_label(bar, labels=[f"{x:.1f}%" for x in percentages], c=colours.labels)

    ax.set_xlabel('Cluster No.', c=colours.labels)
    ax.set_xticks(x_axis)
    ax.set_ylabel('Cluster Sizes', c=colours.labels)
    ax.set_ylim(0, sizes[0]+.2*sizes[0])
    
    ax.tick_params(color=colours.ax, labelcolor=colours.ax)
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_edgecolor(colours.ax)
    for border in ['top', 'right']:
        ax.spines[border].set_visible(False)
    
    fig.savefig(out_path, bbox_inches='tight',
                dpi=config.dpi, transparent=config.transparency)
    plt.close()

