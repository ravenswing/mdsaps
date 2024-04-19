"""
===============================================================================
                      CHAR CHAR CHAR CHAR CHAR CHARTS!!!
===============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import load
from . import plot_config as config

colours = config.colours
sizes = config.sizes


def fes2D(
    fes_path,
    save_path,
    units="A",
    basins=None,
    funnel=None,
    basin_lables=None,
    xlims=None,
    ylims=None,
    labels=["CV1", "CV2"],
):
    """
    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}
    """
    data, lab = load.fes(fes_path, False)
    data[2] = data[2] / 4.184
    if units == "A":
        data[0] = np.multiply(data[0], 10)
        data[1] = np.multiply(data[1], 10)
    cmax = np.amax(data[2][np.isfinite(data[2])]) + 1

    fig = go.Figure(
        data=go.Contour(
            z=data[2],
            x=data[0],  # horizontal axis
            y=data[1],  # vertical axis
            colorscale=colours.map,
            contours=dict(start=0, end=cmax, size=2),
            colorbar=dict(title="Free Energy (kcal/mol)", titleside="right"),
        )
    )
    # format axes
    fig.update_xaxes(
        showline=True,
        linecolor=colours.ax,
        title_text=labels[0],
        linewidth=0.5,
        title_standoff=20,
        ticks="outside",
        minor_ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        linecolor=colours.ax,
        title_text=labels[1],
        linewidth=0.5,
        title_standoff=20,
        ticks="outside",
        minor_ticks="outside",
    )
    if units == "A":
        fig.update_xaxes(dtick=5.0)
        fig.update_yaxes(dtick=2.0)
    if units == "nm":
        fig.update_xaxes(dtick=0.5)
        fig.update_yaxes(dtick=0.2)

    # format the rest of the figure
    fig.update_layout(
        height=1600,
        width=2200,
        title_text="",
        font=dict(color=colours.labels, family="Arial", size=32),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    if xlims:
        fig.update_layout(xaxis_range=xlims)
    if ylims:
        fig.update_layout(yaxis_range=ylims)

    if basins:
        for i, b in enumerate(basins):
            if basin_lables is None:
                fig.add_shape(
                    type="rect",
                    x0=b[0],
                    x1=b[1],
                    y0=b[2],
                    y1=b[3],
                    line_dash="dash",
                    line=dict(color=colours.labels, width=5),
                )
            else:
                fig.add_shape(
                    type="rect",
                    x0=b[0],
                    x1=b[1],
                    y0=b[2],
                    y1=b[3],
                    line_dash="dash",
                    label=dict(text=str(i), font=dict(size=64)),
                    line=dict(color=colours.labels, width=5),
                )

    fig.write_image(save_path, scale=2)


def fes1D(
    fes_path: str,
    cv: str,
    save_path: str,
    cv_label: str = "CV1",
    title: str = "1D Free Energy Surface",
    xlims=None,
    ylims=None,
    walls=None,
    vlines=None,
):
    df = pd.read_table(fes_path, comment="#", sep="\s+", names=[cv, "free", "err"])
    # colvar = load.colvar(f"{wd}/COLVAR")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(df[cv].multiply(10), df.free.divide(4.184), c=colours.default, label="FES")
    # ax.axvline(colvar[cv].iloc[0]*10, ls='dotted', label='Initial', c='xkcd:light gray')
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
        ymax = ylims[1]
    else:
        ymax = df.free.divide(4.184).max() + 2

    if walls:
        for i, wall in enumerate(walls):
            label = "Walls" if i == 0 else None
            ax.fill_between(
                np.linspace(wall[0], wall[1]),
                -2,
                ymax,
                color="k",
                alpha=0.05,
                label=label,
            )

    if vlines:
        for label, position in vlines.items():
            ax.axvline(position, ls="dotted", label=label, c=colours.ax)

    if any([walls, vlines]):
        ax.legend(labelcolor=colours.labels)

    ax.tick_params(color=colours.ax, labelcolor=colours.ax)
    for spine in ax.spines.values():
        spine.set_edgecolor(colours.ax)

    ax.set_xlabel(cv_label, c=colours.labels)
    ax.set_ylabel("Free Energy (kcal/mol)", c=colours.labels)
    ax.set_title(title, c=colours.labels)
    fig.savefig(
        save_path, bbox_inches="tight", dpi=config.dpi, transparent=config.transparency
    )
    plt.close()


def cvs(
    colvar_path,
    save_path,
    cvs,
    cv_labels,
    units="A",
    title="CV Diffusion",
    xlims=None,
    ylims=None,
    mean=False,
    initial=False,
):
    colvar = load.colvar(colvar_path)
    N = len(cvs)

    fig, ax = plt.subplots(1, N, figsize=(8 * N + 2, 6), layout="constrained")

    if N == 1:
        ax = [ax]

    for i, cv in enumerate(cvs):
        ax[i].scatter(
            colvar.time.multiply(0.001),
            colvar[cv].multiply(10),
            c=colours.default,
            s=8,
            alpha=0.4,
        )
        if mean:
            ax[i].axhline(
                colvar[cv].mean() * 10, ls="--", label="Mean", c=colours.highlight
            )
        if initial:
            ax[i].axhline(
                colvar[cv].iloc[0] * 10, ls="dotted", label="Initial", c=colours.ax
            )
        if xlims:
            ax[i].set_xlim(xlims)
        if ylims:
            ax[i].set_ylim(ylims)

        ax[i].set_xlabel("Time (ns)", c=colours.labels, fontsize=sizes.labels)
        label = cv_labels[i] if cv_labels else cv
        ax[i].set_ylabel(label, c=colours.labels, fontsize=sizes.labels)
        ax[i].tick_params(
            axis="both", color=colours.ax, labelcolor=colours.ax, labelsize=sizes.ticks
        )
        for spine in ax[i].spines.values():
            spine.set_edgecolor(colours.ax)

        if any([mean, initial]):
            ax[i].legend(labelcolor=colours.labels, fontsize=sizes.legend)
    fig.suptitle(title, fontsize=sizes.title, c=colours.labels)
    fig.savefig(
        save_path, bbox_inches="tight", dpi=config.dpi, transparent=config.transparency
    )
    plt.close()


def diffusion(DIVS, path_frmt, save_frmt, shape, cvs):
    for cv in cvs:
        fig, ax = plt.subplots(
            shape[0],
            shape[1],
            figsize=(shape[1] * 8, shape[0] * 5),
            sharey=True,
            sharex=True,
        )
        ax = ax.ravel()
        fig.tight_layout(h_pad=4)
        plt.suptitle(f"CV Diffusion - {cvs[cv][0]}")
        plt.subplots_adjust(top=0.94)
        for i, p in enumerate(DIVS):
            colvar = load.colvar(path_frmt.format(p=p))
            ax[i].scatter(colvar.time.multiply(0.001), colvar[cv])
            ax[i].set_title(" ".join(p))
            ax[i].set_xlabel("Simulation Time / ns")
            ax[i].set_ylabel(" / ".join(cvs[cv]))
        fig.savefig(save_frmt.format(cvs[cv][0]), dpi=300, bbox_inches="tight")
        plt.close()


def hills(
    hills_path, save_path, units="A", title="Hill Heights", xlims=None, ylims=None
):
    hills = load.hills(f"{hills_path}")
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    ax.plot([x / 1000 for x in hills[0]], hills[1], c=colours.default)
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)

    ax.set_xlabel("Time (ns)", c=colours.labels, fontsize=sizes.labels)
    ax.set_ylabel("Hill Height", c=colours.labels, fontsize=sizes.labels)
    ax.set_title(f"{title}", c=colours.labels, fontsize=sizes.title)
    ax.tick_params(
        axis="both", color=colours.ax, labelcolor=colours.ax, labelsize=sizes.ticks
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(colours.ax)

    fig.savefig(
        save_path, bbox_inches="tight", dpi=config.dpi, transparent=config.transparency
    )
    plt.close()


def hills_multi(DIVS, path_frmt, save_name, shape):
    fig, ax = plt.subplots(
        shape[0],
        shape[1],
        figsize=(shape[1] * 8, shape[0] * 5),
        sharey=True,
        sharex=True,
    )
    ax = ax.ravel()
    fig.tight_layout(h_pad=4)
    plt.suptitle("Hill Heights")
    plt.subplots_adjust(top=0.94)
    for i, p in enumerate(DIVS):
        data = load.hills(path_frmt.format(p=p))
        ax[i].plot([x / 1000 for x in data[0]], data[1], c=colours.default)
        ax[i].set_title(" ".join(p))
        ax[i].set_xlabel("Simulation Time / ns")
        ax[i].set_ylabel("Hills Heights")
    fig.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.close()


def convergence(
    fes_dir: str,
    fes_format: str,
    cv: str,
    times: list,
    rew_path: str,
    save_path: str,
    cv_label: str = "CV",
    rew_label: str = "Reweight",
    title: str = "Convergence",
    xlims=None,
    ylims=None,
    walls=None,
    vlines=None,
) -> None:
    """Plot convergence of cv"""

    assert len(times) <= 6, "Can only plot 6 FES files."

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, t in enumerate(times):
        df = pd.read_table(
            f"{fes_dir}/{fes_format.format(t)}",
            comment="#",
            sep="\s+",
            names=[cv, "free", "err"],
        )

        ax.plot(
            df[cv].multiply(10),
            df.free.divide(4.184),
            c=colours.rainbow[i],
            label=f"{t} ns",
        )

    rew = pd.read_table(rew_path, comment="#", sep="\s+", names=[cv, "free", "err"])
    ax.plot(
        rew[cv].multiply(10), rew.free.divide(4.184), c="k", ls="--", label=rew_label
    )

    if vlines:
        for label, position in vlines.items():
            ax.axvline(position, ls="dotted", label=label, c=colours.ax)

    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
        ymax = ylims[1]
    else:
        ymax = df.free.divide(4.184).max() + 2

    if walls:
        for i, wall in enumerate(walls):
            label = "Walls" if i == 0 else None
            ax.fill_between(
                np.linspace(wall[0], wall[1]),
                -2,
                ymax,
                color="k",
                alpha=0.05,
                label=label,
            )

    ax.legend(labelcolor=colours.labels, fontsize=sizes.legend)
    ax.tick_params(
        axis="both", color=colours.ax, labelcolor=colours.ax, labelsize=sizes.ticks
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(colours.ax)
    ax.set_xlabel(cv_label, c=colours.labels, fontsize=sizes.labels)
    ax.set_ylabel("Free Energy (kcal/mol)", c=colours.labels, fontsize=sizes.labels)
    ax.set_title(title, c=colours.labels, fontsize=sizes.title)
    fig.savefig(
        save_path, bbox_inches="tight", dpi=config.dpi, transparent=config.transparency
    )
    plt.close()


def dgdt(y, exp_value, ax):
    x = np.linspace(0.0, len(y) * 10, len(y))

    ax.scatter(x, y, c="k", s=8, marker="D", zorder=1)
    ax.plot(x, y, c="k", zorder=2, alpha=0.5)

    ax.axhline(y=exp_value, xmin=0.0, xmax=max(x), c="xkcd:green", ls="--")

    ax.axhspan(
        ymin=exp_value - 2,
        ymax=exp_value + 2,
        xmin=0.0,
        xmax=max(x),
        facecolor="xkcd:green",
        alpha=0.2,
    )
    ax.axhline(y=exp_value + 2, xmin=0.0, xmax=max(x), color="xkcd:green", alpha=0.2)
    ax.axhline(y=exp_value - 2, xmin=0.0, xmax=max(x), color="xkcd:green", alpha=0.2)
    ax.axhspan(
        ymin=exp_value - 3.5,
        ymax=exp_value + 3.5,
        xmin=0.0,
        xmax=max(x),
        facecolor="xkcd:orange",
        alpha=0.2,
    )
    ax.axhline(y=exp_value + 3.5, xmin=0.0, xmax=max(x), color="xkcd:orange", alpha=0.2)
    ax.axhline(y=exp_value - 3.5, xmin=0.0, xmax=max(x), color="xkcd:orange", alpha=0.2)

    ax.set_xlim([-5.0, max(x) + 10])
    ax.set_ylim([-25.0, 5.0])
    ax.grid(alpha=0.3)
