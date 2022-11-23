"""
===============================================================================
                                TRAJECTORY TOOLS
===============================================================================

    - PyTraj based analysis tools for Amber trajectories
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import subprocess
import sys
from glob import glob

sys.path.append('/home/rhys/phd_tools/python_scripts')
import graphics
import load_data as load

import shutil
from math import floor

import MDAnalysis as mda
from MDAnalysis.analysis import diffusionmap, align, rms


DATA_DIR = '/media/rhys/Storage/ampk_metad_all_data'
SYSTS = ['a2b1', 'a2b2']
LIGS = ['A769', 'PF739', 'SC4', 'MT47']

# where to put the plots and other results
SAVE_DIR = '/home/rhys/AMPK/Figures'


def run_sumhills(wd, name, stride=None):
    cmd = f"plumed sum_hills --hills {wd}/HILLS --outfile {wd}/{name}_FES --mintozero"
    if stride is not None:
        cmd += f" --stride {stride}"
    try:
        subprocess.call(cmd, shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def fes_multiplot(cmax=32):
    fig, ax = plt.subplots(4, 2, figsize=(25, 30))
#     fig.tight_layout(h_pad=4)
    plt.suptitle("FES for 500ns Fun-MetaD")
    plt.subplots_adjust(top=0.94, right=0.915)
    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}
    i = 0
    for lig in LIGS:
        j = 0
        for system in SYSTS:
            data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{system}+{lig}_FES', False)
            cmap = graphics.two_cv_contour(data, labels, cmax, ax[i, j], funnel_parms)
            ax[i, j].set_title(f"{system}+{lig}")
            ax[i, j].set_xlabel(f"{labels[0]} / nm")
            ax[i, j].set_ylabel(f"{labels[1]} / nm")
            j += 1
        i += 1
    cax = plt.axes([0.93, 0.11, 0.01, 0.77])
    cbar = plt.colorbar(cmap, cax=cax, aspect=10, ticks=np.arange(0., cmax+1, 2.0))
    cbar.set_label('Free Energy / kcal/mol', fontsize=10)
    fig.savefig(f'{SAVE_DIR}/FES_multi.png', dpi=300, bbox_inches='tight')


def fes_strideplot(wd, name, cmax=32, stride=50, to_use=[0, 1, 2]):
    fig, ax = plt.subplots(1, len(to_use), figsize=(len(to_use*8), 6))
    # fig.tight_layout(h_pad=4)
    plt.suptitle(f"FES Changes over time for {name}")
    plt.subplots_adjust(top=0.85, right=0.915)
    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}
    max_vals = []
    for i in np.arange(len(to_use)):
        data, labels = load.fes(f'{wd}/{name}_FES{to_use[i]}.dat', False)
        data[2] = data[2]/4.184
        max_non_inf = np.amax(data[2][np.isfinite(data[2])])
        max_vals.append(max_non_inf)
        print('VMAX: ', max_non_inf)
    print(f"using: {max(max_vals)}")
    cmax = max(max_vals)+1

    i = 0
    for i in np.arange(len(to_use)):
        data, labels = load.fes(f'{wd}/{name}_FES{to_use[i]}.dat', False)
        cmap = graphics.two_cv_contour(data, labels, cmax, ax[i], funnel_parms)
        ax[i].set_title(f"After {stride*(i+1)} ns")
        ax[i].set_xlabel(f"{labels[0]} / nm")
        ax[i].set_ylabel(f"{labels[1]} / nm")
        i += 1
    cax = plt.axes([0.93, 0.11, 0.01, 0.77])
    cbar = plt.colorbar(cmap, cax=cax, aspect=10,
                        ticks=np.arange(0., cmax, 2.0))
    cbar.set_label('Free Energy / kcal/mol', fontsize=10)
    fig.savefig(f'{SAVE_DIR}/FES_wStride_{name}.png', dpi=300,
                bbox_inches='tight')


if __name__ == "__main__":
    for system in SYSTS:
        for lig in LIGS:
            wd = f"{DATA_DIR}/{system}+{lig}/R1/06-MetaD"
            run_sumhills(wd, f"{system}+{lig}", stride=125000)
            fes_strideplot(wd,
                           f"{system}+{lig}",
                           cmax=32,
                           stride=250)


    # fes_multiplot()
