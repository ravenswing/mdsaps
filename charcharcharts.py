"""
===============================================================================
                            PLOTTING SUPER
===============================================================================

    - Analysis
    - Plotting
    - Calculations
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import plotly.graph_objects as go
from glob import glob
from math import ceil
from plotly.subplots import make_subplots

sys.path.append('/home/rhys/phd_tools/python_scripts')
import load_data as load


import graphics

sys.path.append('/home/rhys/phd_tools/SAPS')
import traj_tools as tt

ANG = "\u212B"

# global colours
CLR = {'ax': 'rgb(69, 69, 69)',
       'ln': 'rgb(116, 62, 122)',
       'f1': 'rgb(251, 218, 230)',
       'f2': 'rgb(255, 241, 194)',
       'A769':  ['#e76f51', '#A23216'],
       'PF739': ['#f4a261', '#994B0B'],
       'SC4':   ['#e9c46a', '#A07918'],
       'MT47':  ['#2a9d8f', '#1E7167'],
       'MK87':  ['#264653', '#13242A']}


def fes_individual(fes_path, save_path, basins=None, funnel=None,
                   labels=['CV1', 'CV2']):
    '''
    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}
    '''
    data, lab = load.fes(fes_path, False)
    data[2] = data[2]/4.184
    cmax = np.amax(data[2][np.isfinite(data[2])])+1

    fig = go.Figure(data=go.Contour(
                        z=data[2],
                        x=data[0],  # horizontal axis
                        y=data[1],  # vertical axis
                        colorscale='RdYlBu',
                        contours=dict(
                           start=0,
                           end=cmax,
                           size=2),
                        colorbar=dict(
                                    title='Free Energy (kcal/mol)',
                                    titleside='right')
                ))
    # format axes
    fig.update_xaxes(showline=True,
                     linecolor=CLR['ax'],
                     title_text=labels[0],
                     linewidth=0.5,
                     title_standoff=20,
                     ticks='outside', minor_ticks='outside')
    fig.update_yaxes(showline=True,
                     linecolor=CLR['ax'],
                     title_text=labels[1],
                     linewidth=0.5,
                     title_standoff=20,
                     ticks='outside', minor_ticks='outside')

    # format the rest of the figure
    fig.update_layout(height=1600, width=2200,
                      title_text="",
                      font=dict(color=CLR['ax'],
                                family='Arial', size=32),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)

    if basins:
        for b in basins:
            fig.add_shape(type="rect",
                          x0=b[0]/10,
                          x1=b[1]/10,
                          y0=b[2]/10,
                          y1=b[3]/10,
                          line_dash='dash',
                          line=dict(color=CLR['ax'], width=5))

    fig.write_image(save_path, scale=2)

    # ax.set_xlabel(
    # ax.set_ylabel(
    # b1 = plt.Rectangle((basins[f'{lig}-bnd'][0]/10, basins[f'{lig}-bnd'][2]/10),

    # cbar.set_label('Free Energy (kcal/mol)') 


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


def all_columns(hdf_path, save_frmt, title='Highlight:',
                labels=None, xlims=None, ylims=None, average=False):

    df = pd.read_hdf(hdf_path, key='df')
    df.columns = df.columns.map('_'.join)
    for col, data in df.items():
        fig, ax = plt.subplots(figsize=(16, 9))
        mean = data.rolling(1000, center=True).mean()
        stdev = data.rolling(1000, center=True).std()

        upr = mean.add(stdev)
        lwr = mean.sub(stdev)

        CLR = 'xkcd:navy'

        ax.plot(df.index*0.001, mean, c=CLR, lw=0.8)
        ax.fill_between(df.index*0.001, upr.values, lwr.values, alpha=0.2)

        if average:
            ax.axhline(y=data.mean(),
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


def hills(DIVS, path_frmt, save_name, shape):
    fig, ax = plt.subplots(shape[0], shape[1],
                            figsize=(shape[1]*8, shape[0]*5),
                            sharey=True, sharex=True)
    ax = ax.ravel()
    fig.tight_layout(h_pad=4)
    plt.suptitle("Hill Heights")
    plt.subplots_adjust(top=0.94)
    for i, p in enumerate(DIVS):
        data = load.hills(path_frmt.format(p=p))
        ax[i].plot([x/1000 for x in data[0]], data[1])
        ax[i].set_title(' '.join(p))
        ax[i].set_xlabel("Simulation Time / ns")
        ax[i].set_ylabel('Hills Heights')
    fig.savefig(save_name, dpi=300, bbox_inches='tight')


def diffusion(DIVS, path_frmt, save_frmt, shape, cvs):
    for cv in list(cvs.keys()):
        fig, ax = plt.subplots(shape[0], shape[1],
                               figsize=(shape[1]*8, shape[0]*5),
                               sharey=True, sharex=True)
        ax = ax.ravel()
        fig.tight_layout(h_pad=4)
        plt.suptitle(f"CV Diffusion - {cvs[cv][0]}")
        plt.subplots_adjust(top=0.94)
        for i, p in enumerate(DIVS):
            colvar = load.colvar(path_frmt.format(p=p))
            ax[i].scatter(colvar.time.multiply(0.001), colvar[cv].multiply(10))
            ax[i].set_title(' '.join(p))
            ax[i].set_xlabel("Simulation Time / ns")
            ax[i].set_ylabel(' / '.join(cvs[cv]))
        fig.savefig(save_frmt.format(cvs[cv][0]), dpi=300, bbox_inches='tight')
