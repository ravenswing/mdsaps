"""
===============================================================================
                                GRAPHICS AND PLOTTING
===============================================================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from math import ceil

from numpy.polynomial import polynomial as P

from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import ticker

from scipy.interpolate import griddata

colours = ['#31859C',   # FS1 & BS1 = RIGHT HAND SIDE
           '#FFC000',   # Tunnel
           '#7030A0']  # FS2 & BS2 = LEFT HAND SIDE


def hills_plot(hills_data, pdb, funnel_side, save_dir):
    """ Plot a simple line graph of HILLS file """
    plt.figure()
    plt.plot([x/1000 for x in hills_data[0]], hills_data[1], label=pdb)
    plt.legend()
    plt.title('{m}  |  {}  |  Hill Heights'.format(funnel_side, m=pdb))
    plt.savefig(f"{save_dir}/{pdb}-{funnel_side}_Heights.png",
                bbox_inches='tight', dpi=300)


def single_diffusion_plots(colvar_data, pdb, funnel_side, num_cvs, save_dir):
    """ plots the diffusion plots for X cvs."""
    if num_cvs == 2:
        x_name = 'X-axis'
        y_name = 'Y-axis'
        plt.figure()
        plt.plot([x/1000 for x in colvar_data[:, 0]],
                 colvar_data[:, 1], label=pdb)
        plt.axhline(0.0, color='k', linestyle='--')
        plt.axhline(4.5, color='k', linestyle='--')
        plt.legend()
        plt.xlabel('Simulation Time / ns')
        plt.ylabel(x_name+' / nm')
        plt.ylim(-0.2, 5.0)
        plt.title('{m}  |  {}  |  Projection on Z - pp.proj '
                  .format(funnel_side, m=pdb))
        plt.savefig('{}/{m}-{f}_proj.png'
                    .format(save_dir, f=funnel_side, m=pdb),
                    bbox_inches='tight', dpi=300)
        plt.figure()
        plt.plot([x/1000 for x in colvar_data[:, 0]],
                 colvar_data[:, 2], label=pdb)
        plt.axhline(0.0, color='k', linestyle='--')
        plt.axhline(2.0, color='k', linestyle='--')
        plt.legend()
        plt.xlabel('Simulation Time / ns')
        plt.ylabel(y_name+' / nm')
        plt.ylim(-0.1, 2.1)
        plt.title('{m}  |  {}  |  Distance from Z - pp.ext'
                  .format(funnel_side, m=pdb))
        plt.savefig('{}/{m}-{f}_ext.png'
                    .format(save_dir, f=funnel_side, m=pdb),
                    bbox_inches='tight', dpi=300)


def two_cv_contour(fes, axes, in_vmax, ax, fp=None, pdb=None,
                   funnel_side=None, name='fun-metad'):
    """ Plot a contour plot for 2 CVs"""

    fes[2] = fes[2]/4.184
    max_non_inf = np.amax(fes[2][np.isfinite(fes[2])])
    #print('VMAX: ', max_non_inf)
    x_name, y_name = axes
    vmax = int(ceil(max_non_inf / 2.0)) * 2 if 'REW' in name else in_vmax

    x, y = np.meshgrid(fes[0], fes[1])

    # iso = round(2*max_non_inf/12)/2

    conts = np.arange(0., vmax+1, 2.0)


#    ax = fig.add_subplot(plot_n, sharex=True, sharey=True)
    CS = ax.contourf(x, y, fes[2], conts, cmap='RdYlBu', antialiased=True)
    ax.contour(x, y, fes[2], conts, colors='k', linewidths=0.5, alpha=0.5,
               antialiased=True)
    if 'REW' not in name and fp is not None:
        f_x = np.linspace(fp['lw'], fp['uw'], 1000)  # lower & upper walls
        sc = fp['sc']   # funnel s-cent
        b = fp['b']     # funnel beta-cent
        f = fp['f']     # funnel wall buffer
        h = fp['h']     # funnel wall width
        f_y = h*(1./(1.+np.exp(b*(f_x-sc))))+f
        ax.plot(f_x, f_y, 'k')
        ax.set_xlim(-0.2, 5.0)
        ax.set_ylim(-0.1, 1.8)
        # ax.set_title('{m}  |  {}'.format(funnel_side, m=pdb))
    else:
        print('?')
        # plt.xlim(-0.2, 5.0)
        # plt.ylim(-0.1, 2.0)
        ax.set_ylabel(y_name+' / nm')
    ax.grid()
    return CS


def OLD_two_cv_contour(fes, pdb, axes, in_vmax, ax):
    ''' Plot a contour plot for 2 CVs'''
    x = fes[:, 0]
    y = fes[:, 1]
    z = fes[:, 2]

    z = np.array(z/4.184)
    z = np.subtract(z, min(z))
    max_non_inf = np.amax(z[np.isfinite(z)])
    print('VMAX: ', max_non_inf)
    x_name, y_name = axes
    # vmax = int(ceil(max_non_inf / 2.0)) * 2 if 'REW' in name else in_vmax
    vmax = in_vmax
    # vmax = 50
    x = np.array([nm*10 for nm in x])
    y = np.array([nm*10 for nm in y])

    xgrid = int(np.sqrt(len(x))-1)
    ygrid = int(np.sqrt(len(y))-1)

    xi = np.linspace(min(x), max(x), xgrid)
    yi = np.linspace(min(y), max(y), ygrid)

    maxz = 0
    for ndx, v in enumerate(z):
        if np.isfinite(z[ndx]):
            if z[ndx] > maxz:
                maxz = z[ndx]

    for ndx, v in enumerate(z):
        if np.isinf(z[ndx]):
            z[ndx] = maxz

    xi, yi = np.meshgrid(xi, yi)
    print(x.shape, y.shape, z.shape, xi.shape, yi.shape,)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # iso = round(2*max_non_inf/12)/2
    conts = np.arange(0.001, vmax+1, 2.0)
    # conts = np.arange(0.001, max(z)+1, 2.0)

    f_x = np.linspace(0.0, 45, 1000)  # funnel lower & upper walls
    sc = 30
    b = 0.15     # funnel beta-cent
    f = 1.5    # funnel wall buffer
    h = 12     # funnel wall width
    f_y = h*(1./(1.+np.exp(b*(f_x-sc))))+f

#    ax = fig.add_subplot(plot_n, sharex=True, sharey=True)
    CS = ax.contourf(xi, yi, zi, conts, cmap='RdYlBu', antialiased=True)
    ax.contour(xi, yi, zi, conts, colors='k', linewidths=0.5, alpha=0.5,
               antialiased=True)
    ax.plot(f_x, f_y, 'k')
    ax.set_xlim(-2.0, 50.0)
    ax.set_ylim(-1.0, 20.0)

    ax.grid()
    return CS


def make_ax(fig, n):
    """ Add axis (for multiplot) """
    x = np.linspace(1, 10)
    y = x * n
    ax = fig.add_subplot(2, 3, n)
    ax.plot(x, y)
    ax.set_title('Times '+str(n))
    return fig


def bubble_plot(csv, size_scale):
    """ Make bubble plot from csv """
    ddg = pd.read_csv(csv, sep=',')
    sizes = (ddg["bonds"] + 1)*size_scale
    sns.scatterplot(x='weight', y='fs2', s=sizes, data=ddg,)
    sns.scatterplot(x='weight', y='fs1', s=sizes, data=ddg, legend='brief')
    plt.xlabel('Molecular Weight / Da')
    plt.ylabel('Deviation from Experimental $\Delta$G / kcal/mol')
    plt.grid(alpha=0.5, zorder=1)
    plt.savefig(csv+'_bubble.png', dpi=300, transparent=True)


def ddg_scatter(csv, mode):
    """ Make custom scatter from ddG data """
    ddg = pd.read_csv(csv, sep=',')
    plt.figure()
    line1 = P.polyfit(x=ddg['weight'], y=ddg['fs1'], deg=1)
    f1 = P.Polynomial(line1)
    line2 = P.polyfit(x=ddg['weight'], y=ddg['fs2'], deg=1)
    f2 = P.Polynomial(line2)
    line3 = P.polyfit(x=pd.concat([ddg['weight'], ddg['weight']]),
                      y=pd.concat([ddg['fs1'], ddg['fs2']]), deg=1)
    f3 = P.Polynomial(line3)
    x = np.linspace(150, 550, 50)
    plt.hlines(y=0, xmin=150, xmax=550, colors='xkcd:green', linewidth=2.5)

    if mode == 1:
        plt.errorbar(x='weight', y='fs2', yerr=2.0, data=ddg,
                     fmt='o', capsize=5, c=colours[2], label='FS2')
        plt.errorbar(x='weight', y='fs1', yerr=2.0, data=ddg,
                     fmt='o', capsize=5, c=colours[0], label='FS1')
    if mode == 2:
        plt.scatter(x='weight', y='fs2', data=ddg,
                    marker='D', s=6, c=colours[2], label='FS2', zorder=2)
        plt.scatter(x='weight', y='fs1', data=ddg,
                    marker='D', s=6, c=colours[0], label='FS1', zorder=3)
        plt.axhspan(-2, 2, facecolor='xkcd:green', alpha=0.2, zorder=1)

    plt.plot(x, f1(x), '--', c=colours[0], alpha=0.5)
    plt.plot(x, f2(x), '--', c=colours[2], alpha=0.5)
    plt.plot(x, f3(x), 'k', label='Combined Trend')
    plt.xlim([150, 550])
    plt.ylim([-1., 15.])
    plt.xlabel('Molecular Weight / Da')
    plt.ylabel('Deviation from Experimental $\Delta$G / kcal/mol')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.savefig(csv.split('.')[0]+str(mode)+'_scatter.png',
                dpi=300, transparent=True)


def convergence(fes_dir, ts_list, ax):
    """ Plot convergence of cv """
    lin_cols = ['xkcd:light red', 'xkcd:light orange', 'xkcd:light green',
                'xkcd:light cyan', 'xkcd:ocean blue']
    init_file = '{}/fes_{}.dat'.format(fes_dir, int(ts_list[0]/10))
    conv_data = pd.concat([df[df.cv != "#!"] for df in
                          pd.read_csv(init_file,
                                      delim_whitespace=True,
                                      names=['cv', str(ts_list[0])],
                                      skiprows=5,
                                      chunksize=1000)])
    for timestamp in ts_list[1:]:
        fes_file = '{}/fes_{}.dat'.format(fes_dir, int(timestamp/10))
        fes_data = pd.concat([df[df.cv != "#!"] for df in
                             pd.read_csv(fes_file,
                                         delim_whitespace=True,
                                         names=['cv', str(timestamp)],
                                         skiprows=5, chunksize=1000)])
        conv_data = pd.merge(conv_data, fes_data, on='cv')
    for i in np.arange(len(ts_list)):
        ax.plot(conv_data['cv'],
                [y/4.184 for y in conv_data[str(ts_list[i])]],
                c=lin_cols[i], label=str(ts_list[i])+' ns')

    nm = re.split('/|_', fes_dir)
    rew_file = '{p}_{f}/{p}-{f}_{c}.fes'.format(p=nm[0], f=nm[1], c=nm[-1])
    rew_data = pd.read_csv(rew_file,
                           delim_whitespace=True,
                           names=['rx', 'ry'],
                           skiprows=5)
    ax.plot(rew_data['rx'], [y/4.184 for y in rew_data['ry']], 'k')
    if 'proj' in fes_dir:
        ax.set_xlim([-0.3, 5.0])
        ax.set_xticks(np.arange(6))
    else:
        ax.set_xlim([-0.1, 1.7])
        ax.set_xticks(np.linspace(0., 1.5, num=4))
    ax.set_ylim([0, 20.])
    ax.grid(alpha=0.5)


def diffusion(colv_data, cv, ax, lin_col):
    """ Plot diffusion of CV """
    colv_data['pp.'+cv] = colv_data['pp.'+cv].astype(float)
    colv_data['mean'] = colv_data['pp.'+cv].rolling(5000, center=True).mean()
    y1 = colv_data['pp.'+cv].values
    y2 = colv_data['mean'].values
    x = np.arange(len(y1))*0.002
    ax.plot(x, y1, c=lin_col, alpha=0.3, lw=0.5)
    ax.plot(x, y2, c='k', alpha=1., lw=1.)
    if cv == 'proj':
        ax.set_ylim([-0.3, 5.0])
        ax.set_yticks(np.arange(6))
        ax.axhline(y=3.0, xmax=500., c='k', alpha=0.5, lw=1., ls='--')
        ax.axhline(y=0.7, xmax=500., c='k', alpha=0.5, lw=1., ls='--')
    else:
        ax.set_ylim([-0.1, 1.7])
        ax.set_yticks(np.linspace(0., 1.5, num=4))
    ax.set_xlim([0, 500.])
    ax.set_xticks(np.linspace(0., 500., num=6))
    ax.grid(alpha=0.3)


def dgdt(y, exp_value, ax):

    x = np.linspace(0., len(y)*10, len(y))

    ax.scatter(x, y, c='k', s=8, marker='D', zorder=1)
    ax.plot(x, y, c='k', zorder=2, alpha=.5)

    ax.axhline(y=exp_value, xmin=0., xmax=max(x), c='xkcd:green', ls='--')

    ax.axhspan(ymin=exp_value-2, ymax=exp_value+2, xmin=0., xmax=max(x),
               facecolor='xkcd:green', alpha=0.2)
    ax.axhline(y=exp_value+2, xmin=0., xmax=max(x),
               color='xkcd:green', alpha=0.2)
    ax.axhline(y=exp_value-2, xmin=0., xmax=max(x),
               color='xkcd:green', alpha=0.2)
    ax.axhspan(ymin=exp_value-3.5, ymax=exp_value+3.5, xmin=0., xmax=max(x),
               facecolor='xkcd:orange', alpha=0.2)
    ax.axhline(y=exp_value+3.5, xmin=0., xmax=max(x),
               color='xkcd:orange', alpha=0.2)
    ax.axhline(y=exp_value-3.5, xmin=0., xmax=max(x),
               color='xkcd:orange', alpha=0.2)

    ax.set_xlim([-5., max(x)+10])
    ax.set_ylim([-25., 5.0])
    ax.grid(alpha=0.3)


def xvg_line(xvg_data, ax, col, line='solid', rollavg=True, label=None):
    head = xvg_data.columns.values.tolist()
    xvg_data['mean'] = xvg_data[head[1]].rolling(500, center=True).mean()
    y1 = xvg_data[head[1]].values*10
    y2 = xvg_data['mean'].values*10
    x = xvg_data[head[0]]/1000
    ax.plot(x, y1, c=col, alpha=0.3, lw=0.5)
    if label:
        ax.plot(x, y2, c=col, alpha=1., lw=1., ls=line, label=label)
    else:
        ax.plot(x, y2, c=col, alpha=1., lw=1., ls=line)
    ax.grid(alpha=0.3)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=False,
                   labeltop=False, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    # original rotation = -30
    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, min_cutoff=None,
                     max_cutoff=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if min_cutoff is not None and data[i, j] < min_cutoff:
                continue
            elif max_cutoff is not None and data[i, j] > max_cutoff:
                continue
            else:
                kw.update(color=textcolors[int(im.norm(data[i, j])
                                           > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

    return texts


def rmsf(rmsf_data, seq, grid=None, offset=None):
    '''
    to plot the rmsf data - specifically for B-sheet and lamellars
        rmsf_data       output from load_data.xvg
        seq             Sequence: list of residue names/letters
        grid            dimensions of heatmap grid is displaying as such
        offset          if anti-parallel, number of cells to displace each
                        alternating row by
    '''
    # adjust cmap so minimum value is perfectly white
    old_cmap = cm.get_cmap('YlGnBu', 256)
    newcolors = old_cmap(np.linspace(0, 1, 256))
    white = np.array([256/256, 256/256, 256/256, 1])
    newcolors[0, :] = white
    custom_cmap = ListedColormap(newcolors)
    # max_cmap_val = 12.
    max_cmap_val = 20.

    # extract data from loaded xvg pd DataFrame
    head = rmsf_data.columns.values.tolist()
    rmsf = np.array(rmsf_data[head[1]].tolist())*10

    # extract molecule name from sequence
    out_name = ''.join([seq[-1], seq[-2], seq[1], seq[0]])

    # plots a bar or the RMSF per residue
    if grid is None:
        print('2D plot')

    elif offset is None:
        print('Plotting for parallel arrangement')
        assert len(grid) == 2
        assert rmsf_data.shape[0] == grid[0] * grid[1]

        # reshape to grid dimensions
        rmsf = np.reshape(rmsf, (grid[0], grid[1]))

        fig, ax = plt.subplots(figsize=(11, 3.5))

        im, cbar = heatmap(rmsf, [x+1 for x in list(np.arange(grid[0]))], seq,
                           ax=ax, cmap=custom_cmap, cbarlabel="RMSF $\AA$",
                           vmin=0., vmax=max_cmap_val)

        annotate_heatmap(im, valfmt="{x:.1f}", min_cutoff=-1.)

        for s in np.arange(len(seq)):
            im.axes.text(s, -1, seq[s], horizontalalignment='center')

        out_name += '_para'

    else:
        print('Plotting for antiparallel arrangement')
        assert len(grid) == 2
        assert isinstance(offset, int)

        seq = [' '] * offset + seq

        # establish a grid of negatives that will not be shown
        base = np.ones((grid[0], grid[1]+offset))*-10.
        for i in range(grid[0]):
            if i % 2 == 0:
                base[i, offset:] = rmsf[i*grid[1]:(i+1)*grid[1]]
            else:
                base[i, :-offset] = np.flip(rmsf[i*grid[1]:(i+1)*grid[1]])

        fig, ax = plt.subplots(figsize=(12, 5))

        im, cbar = heatmap(base, [x+1 for x in list(np.arange(6))], seq,
                           ax=ax, cmap=custom_cmap, cbarlabel="RMSF $\AA$",
                           vmin=0., vmax=max_cmap_val)

        annotate_heatmap(im, valfmt="{x:.1f}", min_cutoff=-1.)

        for s in np.arange(len(seq)):
            im.axes.text(s, -1, seq[s], horizontalalignment='center')
            im.axes.text(s-1, grid[0], seq[-s], horizontalalignment='center')

        out_name += '_anti'

    fig.tight_layout()

    return fig, out_name

    # fig.savefig('./RMSF_{}.png'.format(out_name), dpi=300,
                # bbox_inches='tight', transparent=True)
