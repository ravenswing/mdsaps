"""
===============================================================================
                    Gromacs Organisation n' Analysis Tools
===============================================================================

    - sumhills
    - 
    - 
"""

import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import subprocess
import shutil
import sys
from glob import glob
from math import ceil, floor
import MDAnalysis as mda
from MDAnalysis.analysis import diffusionmap, align, rms

sys.path.append('/home/rhys/phd_tools/python_scripts')
import graphics
import load_data as load

sys.path.append('/home/rhys/phd_tools/SAPS')
import traj_tools as tt


def run_sumhills(wd, name, stride=None):
    cmd = (f"plumed sum_hills --hills {wd}/HILLS "
           f"--outfile {wd}/{name}_FES --mintozero")
    if stride is not None:
        cmd += f" --stride {stride}"
    try:
        subprocess.run(cmd,
                       shell=True,
                       check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def gismo_traj(wd, in_path, out_path, tpr='prod.tpr', ndx='i.ndx'):
    """ cutdown the trajectories using Gromacs trjconv ready for GISMO """
    # call gmx trjconv with -dt 100 to cut down the trajectory
    cmd = ("echo Backbone Protein_LIG | gmx_mpi trjconv "
           f"-s {wd}/{tpr} "
           f"-f {wd}/{in_path} "
           f"-o {wd}/{out_path} "
           f"-n {wd}/{ndx} "
           "-fit rot+trans "
           "-dt 100 ")
    try:
        subprocess.run(cmd,
                       shell=True,
                       check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def gismo_colvar(wd, in_colvar='COLVAR', out_colvar='COLVAR_GISMO'):
    """ combine old and reweighted colvars """
    # Load in the original COLVAR
    old_col = load.colvar(f"{wd}/{in_colvar}", 'as_pandas')

    # Cutdown old COLVAR to match trajectories by selecting every 5th line
    old_col = old_col.iloc[::5, :]
    # Add every 10th line (and the second line) for GISMO colvar
    gis_col = old_col.iloc[:2, :]
    gis_col = gis_col.append(old_col.iloc[10::10, :], ignore_index=True)

    # Define path for the output GISMO COLVAR file
    gismo_col_path = f"{wd}/{out_colvar}"
    # Add the header line to this new COLVAR
    with open(gismo_col_path, 'w') as f:
        f.write("#! FIELDS "+" ".join(list(gis_col.columns.values))+"\n")
    # Save the cutdown GISMO COLVAR
    gis_col.to_csv(gismo_col_path, sep=" ",
                   header=False, index=False, mode='a')
    print(f"Successfully converted {in_colvar} to {out_colvar}.")
