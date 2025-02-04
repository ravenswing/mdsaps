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
    cvs=None,
    units="A",
    title: str = "2D Free Energy Surface",
    labels=None,
    xlims=None,
    ylims=None,
    dticks=(None, None),
    funnel=None,
    basins=None,
    basin_labels=None,
    contour_width=None,
    contour_max=None,
):
    """
    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}
    data, lab = load.fes(fes_path, False)
    data[2] = data[2] / 4.184
    if units == "A":
        data[0] = np.multiply(data[0], 10)
        data[1] = np.multiply(data[1], 10)
    """

    # For COLVARS with more than 2 CVs, allow the selection of 2.
    if cvs is not None:
        assert len(cvs) == 2, "Must provide two CVs for plotting."
        fes, _ = load.fes(fes_path)
    else:
        fes, cvs = load.fes(fes_path)
        assert (
            len(cvs) == 2
        ), "Number of CVs found in FES not equal to 2. Please specify with cvs."

    labels = labels if labels else cvs
    # default value for fun-metaD etc. = 2
    if contour_width:
        if contour_max:
            cmax = contour_max
        else:
            z_finite = np.isfinite(fes.free)
            cmax = np.amax(fes.free[z_finite].divide(4.184)) + 1
        contours = dict(start=0, end=cmax, size=contour_width)
    else:
        contours = None

    # TODO - Add units multiply!!!
    fig = go.Figure(
        data=go.Contour(
            z=fes.free.divide(4.184),
            x=fes[cvs[0]].multiply(10),  # horizontal axis
            y=fes[cvs[1]].multiply(10),  # horizontal axis
            colorscale=colours.map,
            contours=contours,
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

    x_tickstep, y_tickstep = dticks
    if x_tickstep:
        fig.update_xaxes(dtick=x_tickstep)
    else:
        fig.update_xaxes(dtick=5.0)
    if y_tickstep:
        fig.update_yaxes(dtick=y_tickstep)
    else:
        fig.update_yaxes(dtick=2.0)

    # format the rest of the figure
    fig.update_layout(
        height=1600,
        width=2200,
        title_text=title,
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
            if basin_labels is None:
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
                size = 64 if len(basin_labels[i]) <= 4 else 32
                fig.add_shape(
                    type="rect",
                    x0=b[0],
                    x1=b[1],
                    y0=b[2],
                    y1=b[3],
                    line_dash="dash",
                    label=dict(text=basin_labels[i], font=dict(size=size)),
                    line=dict(color=colours.labels, width=5),
                )

    fig.write_image(save_path, scale=2)


def fes1D(
    fes_path: str,
    save_path: str,
    label=None,
    title: str = "1D Free Energy Surface",
    xlims=None,
    ylims=None,
    walls=None,
    vlines=None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")
        save = True
    else:
        save = False

    fes, cvs = load.fes(fes_path)
    assert len(cvs) == 1, "ERROR: More than 1 CV in FES. Please provide a 1D FES."
    cv = cvs[0]
    label = label if label else cv

    ax.plot(
        fes[cv].multiply(10), fes.free.divide(4.184), c=colours.default, label="FES"
    )
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
        ymax = ylims[1]
    else:
        ymax = fes.free.divide(4.184).max() + 2

    if walls:
        for i, wall in enumerate(walls):
            wall_label = "Walls" if i == 0 else None
            ax.fill_between(
                np.linspace(wall[0], wall[1]),
                -2,
                ymax,
                color="k",
                alpha=0.05,
                label=wall_label,
            )

    if vlines:
        for legend_label, position in vlines.items():
            ax.axvline(position, ls="dotted", label=legend_label, c=colours.ax)

    if any([walls, vlines]):
        ax.legend(labelcolor=colours.labels)

    ax.tick_params(color=colours.ax, labelcolor=colours.ax)
    for spine in ax.spines.values():
        spine.set_edgecolor(colours.ax)

    ax.set_xlabel(label, c=colours.labels)
    ax.set_ylabel("Free Energy (kcal/mol)", c=colours.labels)
    ax.set_title(title, c=colours.labels)
    if save:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=config.dpi,
            transparent=config.transparency,
        )
    else:
        return ax
    plt.close()


def cvs(
    colvar_path,
    cvs,
    save_path="cvs.png",
    units="A",
    title="CV Diffusion",
    cv_labels=None,
    xlims=None,
    ylims=None,
    mean=False,
    initial=False,
    ax=None,
):
    colvar = load.colvar(colvar_path)
    N = len(cvs)

    if ax is None:
        fig, ax = plt.subplots(1, N, figsize=(8 * N + 2, 6), layout="constrained")
        save = True
    else:
        assert len(ax) == N, "Number of CVs does not match the shape of supplied axis"
        save = False

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
    if save:
        fig.suptitle(title, fontsize=sizes.title, c=colours.labels)
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=config.dpi,
            transparent=config.transparency,
        )
    else:
        return ax
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
    hills_path,
    save_path,
    units="A",
    title="Hill Heights",
    xlims=None,
    ylims=None,
    ax=None,
):
    hills = load.hills(f"{hills_path}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        save = True
    else:
        save = False

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

    if save:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=config.dpi,
            transparent=config.transparency,
        )
    else:
        return ax


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
    save_path: str = "conv_plot.png",
    cv_label: str = "CV",
    rew_label: str = "Reweight",
    title: str = "Convergence",
    xlims=None,
    ylims=None,
    walls=None,
    vlines=None,
    ax=None,
) -> None:
    """Plot convergence of cv"""

    assert len(times) <= 6, "Can only plot 6 FES files."

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        save = True
    else:
        save = False

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

    # rew = pd.read_table(rew_path, comment="#", sep="\s+", names=[cv, "free", "err"])
    rew = pd.read_table(rew_path, comment="#", sep="\s+", names=[cv, "free"])
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

    if save:
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=config.dpi,
            transparent=config.transparency,
        )
    else:
        return ax
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
