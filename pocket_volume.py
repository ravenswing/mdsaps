import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import chain
import subprocess
import sys
import os
import re
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from MDAnalysis.analysis import align, rms
import load
import traj_tools as tt

systems = {'a2b1': ['A769', 'PF739', 'SC4', 'MT47', 'MK87'],
           'a2b2': ['A769', 'PF739', 'SC4', 'MT47', 'MK87']}

DATA_DIR = '/home/rhys/Storage/ampk_metad_all_data'


def aligned_pdb(wd, ref_path):
    u = tt._init_universe(f"{wd}/md_dry.pdb")
    protein = u.select_atoms("protein or resname S2P")
    with mda.Writer(f'{wd}/tmp_prot.pdb', protein.n_atoms) as W:
        for ts in u.trajectory:
            W.write(protein)

    mobile = tt._init_universe(f'{wd}/tmp_prot.pdb')
    ref = tt._init_universe(ref_path)
    aligner = align.AlignTraj(mobile, ref, select='backbone',
                              filename=f'{wd}/aligned.pdb').run()


def aligned_dcd(wd, xtc_name, ref_path):
    # ADD NFRAMES ARGUMENT AN LINSPACE FOR FRAME ITERATION!
    u = tt._init_universe([f"{wd}/md_dry.pdb", f"{wd}/{xtc_name}"])
    protein = u.select_atoms("protein or resname S2P")
    with mda.Writer(f'{wd}/tmp_prot.xtc', protein.n_atoms) as W:
        for ts in u.trajectory[::5]:
            W.write(protein)

    mobile = tt._init_universe([f'{wd}/aligned.pdb', f'{wd}/tmp_prot.xtc'])
    ref = tt._init_universe(ref_path)
    aligner = align.AlignTraj(mobile, ref, select='backbone',
                              filename=f'{wd}/aligned.dcd').run()


def pocket_select(wd, out_name):
    mpck_cmd = ("mdpocket --trajectory_file aligned.dcd --trajectory_format dcd "
                f"-f aligned.pdb -o {out_name} -n 3.0")
    try:
        subprocess.run(mpck_cmd, cwd=wd, shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


for method in ['fun-metaD']:
    for system in systems.keys():
        out_dir = f"/media/rhys/Storage/ampk_metad_all_data/analysis_data/mdpocket/{system}"
        ref = f"/home/rhys/Storage/ampk_metad_all_data/super_ref/{system}.pdb"
        for pdb in systems[system]:
            for rep in ['R1', 'R2', 'R3', 'R4']:
                wd = f"{DATA_DIR}/{method}/{system}+{pdb}/{rep}"
                aligned_pdb(wd, ref)
                aligned_dcd(wd, f"{system}+{pdb}_{rep}_GISMO.xtc", ref)
                try:
                    subprocess.run('rm tmp_*', cwd=wd, shell=True, check=True)
                except subprocess.CalledProcessError as error:
                    print('Error code:', error.returncode,
                        '. Output:', error.output.decode("utf-8"))
                pocket_select(wd, f"{system}+{pdb}_{rep}")
                try:
                    subprocess.run(f'cp *_freq_iso_* {out_dir}', cwd=wd, shell=True, check=True)
                except subprocess.CalledProcessError as error:
                    print('Error code:', error.returncode,
                        '. Output:', error.output.decode("utf-8"))
                try:
                    subprocess.run(f'cp *_atom_pdens* {out_dir}', cwd=wd, shell=True, check=True)
                except subprocess.CalledProcessError as error:
                    print('Error code:', error.returncode,
                        '. Output:', error.output.decode("utf-8"))
