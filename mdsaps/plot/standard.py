"""
===============================================================================
                      CHAR CHAR CHAR CHAR CHAR CHARTS!!!
===============================================================================
"""

import matplotlib.pyplot as plt
import pandas as pd

from . import plot_config as config
from ..tools import usym

colours = config.colours
sizes = config.sizes


def rolling_mean(
    x,
    y,
    save_path,
    labels,
    title="Rolling Mean Plot",
    raw_data=None,
    xlims=None,
    ylims=None,
    mean=False,
    initial=False,
    window=1000,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")

    if raw_data == "scatter":
        ax.scatter(x, y, c=colours.default, s=8, alpha=0.2, label="Raw Data")
    elif raw_data == "line":
        ax.plot(x, y, c=colours.default, lw=0.5, alpha=0.2, label="Raw Data")

    rolling_average = y.rolling(window, center=True).mean()
    rolling_stdev = y.rolling(window, center=True).std()

    upr = rolling_average.add(rolling_stdev)
    lwr = rolling_average.sub(rolling_stdev)

    ax.plot(x, rolling_average, c=colours.default, lw=0.8, label="Rolling Mean")
    ax.fill_between(x, upr.values, lwr.values, alpha=0.3, label="Std. Dev.")

    if mean:
        ax.axhline(
            y.mean(),
            ls="--",
            label=f"Mean = {y.mean():.1f} {usym('pm')} {y.std():.1f}",
            c=colours.highlight,
        )
    if initial:
        ax.axhline(y.iloc[0], ls="dotted", label="Initial", c=colours.ax)
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)

    ax.set_xlabel(labels[0], c=colours.labels, fontsize=sizes.labels)
    ax.set_ylabel(labels[1], c=colours.labels, fontsize=sizes.labels)
    ax.tick_params(
        axis="both", color=colours.ax, labelcolor=colours.ax, labelsize=sizes.ticks
    )
    for axis in ["bottom", "left"]:
        ax.spines[axis].set_edgecolor(colours.ax)
    for border in ["top", "right"]:
        ax.spines[border].set_visible(False)

    if any([mean, initial]):
        ax.legend(labelcolor=colours.labels, fontsize=sizes.legend)
    fig.suptitle(title, fontsize=sizes.title, c=colours.labels)
    fig.savefig(
        save_path, bbox_inches="tight", dpi=config.dpi, transparent=config.transparency
    )
    plt.close()


def multiplot(
    hdf_path,
    DIVS,
    save_frmt,
    title_frmt,
    plot_index=0,
    labels=None,
    xlims=None,
    ylims=None,
    average=False,
):
    data = pd.read_hdf(hdf_path, key="df")

    plots = DIVS.pop(plot_index)
    print(plots)
    print(DIVS)

    for p, plot in enumerate(plots):
        # initiate plots and titles
        fig, ax = plt.subplots(
            len(DIVS[0]), len(DIVS[1]), figsize=(len(DIVS[1]) * 8, len(DIVS[0]) * 5)
        )
        plt.suptitle(title_frmt.format(plot))
        plt.subplots_adjust(top=0.95)
        # add the plots to the axes
        for i, t1 in enumerate(DIVS[0]):
            for j, t2 in enumerate(DIVS[1]):
                if plot_index == 0:
                    df = data[plot][t1][t2]
                elif plot_index == 1:
                    df = data[t1][plot][t2]
                elif plot_index == 2:
                    df = data[t1][t2][plot]
                else:
                    print("UNSUPPORTED PLOT INDEX")
                mean = pd.Series(df).rolling(1000, center=True).mean()
                stdev = pd.Series(df).rolling(1000, center=True).std()

                upr = mean.add(stdev)
                lwr = mean.sub(stdev)

                col = "xkcd:navy"

                ax[i, j].plot(
                    df.index * 0.001, mean, c=col, lw=0.8, label=f"{t1} - {t2}"
                )
                ax[i, j].fill_between(
                    df.index * 0.001, upr.values, lwr.values, alpha=0.3
                )

                if average:
                    ax[i, j].axhline(y=df.mean(), c="k", alpha=0.5, lw=1.5, ls="--")

                # ax[i, j].set_title(f'{t1} - {t2}')
                ax[i, j].legend()
                if xlims:
                    ax[i, j].set_xlim(xlims)
                if ylims:
                    ax[i, j].set_ylim(ylims)
                if labels and i == 1:
                    ax[i, j].set_xlabel(labels[0])
                if labels and j == 0:
                    ax[i, j].set_ylabel(labels[1])
        fig.savefig(save_frmt.format(plot), dpi=450, bbox_inches="tight")
        plt.close()


def all_columns(
    hdf_path,
    save_frmt,
    title="Highlight:",
    labels=None,
    xlims=None,
    ylims=None,
    hline=False,
):
    df = pd.read_hdf(hdf_path, key="df")
    df.columns = df.columns.map("_".join)
    for col, data in df.items():
        fig, ax = plt.subplots(figsize=(12, 6.75))
        mean = data.rolling(1000, center=True).mean()
        stdev = data.rolling(1000, center=True).std()

        upr = mean.add(stdev)
        lwr = mean.sub(stdev)

        ax.plot(df.index * 0.001, mean, c=c, lw=0.8)
        ax.fill_between(df.index * 0.001, upr.values, lwr.values, alpha=0.2)

        if hline:
            if hline == "average":
                ax.axhline(y=data.mean(), c="k", alpha=0.5, lw=1.5, ls="--")
            else:
                ax.axhline(y=hline, c="k", alpha=0.5, lw=1.5, ls="--")
        if xlims:
            ax.set_xlim(xlims)
        if ylims:
            ax.set_ylim(ylims)
        if labels:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        ax.set_title(f"{title} {' '.join(col.split('_'))}")
        fig.savefig(save_frmt.format(col), dpi=450, bbox_inches="tight")
        plt.close()


def xvg_line(xvg_data, ax, col, line="solid", label=None):
    head = xvg_data.columns.values.tolist()
    xvg_data["mean"] = xvg_data[head[1]].rolling(500, center=True).mean()
    y1 = xvg_data[head[1]].values * 10
    y2 = xvg_data["mean"].values * 10
    x = xvg_data[head[0]] / 1000
    ax.plot(x, y1, c=col, alpha=0.3, lw=0.5)
    ax.plot(x, y2, c=col, alpha=1.0, lw=1.0, ls=line, label=label)
    ax.grid(alpha=0.3)
