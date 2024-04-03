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
from ..tools import usym
from . import plot_config as config

clr = config.colour_dict
c = config.default_colour


def multiplot(hdf_path, DIVS, save_frmt, title_frmt,
              plot_index=0, labels=None, xlims=None, ylims=None,
              average=False):

    data = pd.read_hdf(hdf_path, key='df')

    plots = DIVS.pop(plot_index)
    print(plots)
    print(DIVS)

    for p, plot in enumerate(plots):
        # initiate plots and titles
        fig, ax = plt.subplots(len(DIVS[0]), len(DIVS[1]),
                               figsize=(len(DIVS[1])*8, len(DIVS[0])*5))
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
                    print('UNSUPPORTED PLOT INDEX')
                mean = pd.Series(df).rolling(1000, center=True).mean()
                stdev = pd.Series(df).rolling(1000, center=True).std()

                upr = mean.add(stdev)
                lwr = mean.sub(stdev)

                col = 'xkcd:navy'

                ax[i, j].plot(df.index*0.001, mean, c=col,
                              lw=0.8, label=f'{t1} - {t2}')
                ax[i, j].fill_between(df.index*0.001, upr.values, lwr.values,
                                      alpha=0.3)

                if average:
                    ax[i, j].axhline(y=df.mean(),
                                     c='k', alpha=0.5, lw=1.5, ls='--')

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
        fig.savefig(save_frmt.format(plot), dpi=450, bbox_inches='tight')
        plt.close()


def all_columns(hdf_path, save_frmt, title='Highlight:',
                labels=None, xlims=None, ylims=None, hline=False):

    df = pd.read_hdf(hdf_path, key='df')
    df.columns = df.columns.map('_'.join)
    for col, data in df.items():
        fig, ax = plt.subplots(figsize=(12, 6.75))
        mean = data.rolling(1000, center=True).mean()
        stdev = data.rolling(1000, center=True).std()

        upr = mean.add(stdev)
        lwr = mean.sub(stdev)

        ax.plot(df.index*0.001, mean, c=c, lw=0.8)
        ax.fill_between(df.index*0.001, upr.values, lwr.values, alpha=0.2)

        if hline:
            if hline == 'average':
                ax.axhline(y=data.mean(),
                           c='k', alpha=0.5, lw=1.5, ls='--')
            else:
                ax.axhline(y=hline,
                           c='k', alpha=0.5, lw=1.5, ls='--')
        if xlims:
            ax.set_xlim(xlims)
        if ylims:
            ax.set_ylim(ylims)
        if labels:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        ax.set_title(f"{title} {' '.join(col.split('_'))}")
        fig.savefig(save_frmt.format(col), dpi=450, bbox_inches='tight')
        plt.close()


def xvg_line(xvg_data, ax, col, line='solid', rollavg=True, label=None):
    head = xvg_data.columns.values.tolist()
    xvg_data['mean'] = xvg_data[head[1]].rolling(500, center=True).mean()
    y1 = xvg_data[head[1]].values*10
    y2 = xvg_data['mean'].values*10
    x = xvg_data[head[0]]/1000
    ax.plot(x, y1, c=col, alpha=0.3, lw=0.5)
    ax.plot(x, y2, c=col, alpha=1., lw=1., ls=line, label=label)
    ax.grid(alpha=0.3)
