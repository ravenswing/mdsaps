"""
===============================================================================
                                TRAJECTORY TOOLS
===============================================================================
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parmed as pmd
import pytraj as pt
from glob import glob
from itertools import chain
import subprocess
import sys
import os


def _load_structure(in_str):
    # .pdb only req. 1 input
    if isinstance(in_str, str) and '.pdb' in in_str:
        return pt.load(in_str)
    # .r and same named .top
    elif isinstance(in_str, str):
        return pt.load(in_str, in_str.split('.')[0]+'.top')
    # explicitly specified crd and top
    elif isinstance(in_str, list) and len(in_str) == 2:
        return pt.load(in_str[0], in_str[1])
    # for references, can just be int i.e. frame no.
    elif isinstance(in_str, int):
        return in_str
    # if not any of the above raise an error
    else:
        raise ValueError("Structure not recognised")


def align(in_str, ref_str, out_str, aln_mask='@CA,C,N,O', strip_mask=None):
    # load the initial structure
    to_align = _load_structure(in_str)
    ref = _load_structure(ref_str)
    # run the alignment
    aligned = pt.align(to_align, mask=aln_mask, ref=ref)
    aligned = aligned.autoimage()
    # if strip is required, perform the strip
    if strip_mask is not None:
        aligned = aligned.strip(strip_mask)
    # write the new str
    pt.write_traj(out_str, aligned, overwrite=True)


def cut_traj(trj_path, top, out_path, denom=100, split=False, strip_mask=None):
    full_trj = pt.iterload(trj_path, top)
    full_trj = full_trj.autoimage()
    print(f'Loaded trajectory: {trj_path}')
    if not split:
        start_point = 1
        print(f'NOT cutting traj so starting from {start_point}')
        N = int(full_trj.n_frames/denom)
    else:
        start_point = int(full_trj.n_frames/2)
        print(f'CUTTING traj, starting from {start_point}')
        N = int(start_point/denom)
    print(f'Writing {N} frames')
    frames = np.linspace(start_point, full_trj.n_frames, num=N, dtype=int)-1
    # if strip is required, perform the strip
    if strip_mask is not None:
        full_trj = full_trj.strip(strip_mask)
    pt.write_traj(out_path, full_trj, frame_indices=frames, overwrite=True)
    print(f'Saved new trajectory: {out_path}')


def measure_rmsd(trj_path, top_path, ref_str, rmsd_mask, aln_mask='@CA,C,N,O'):
    # load the trajectory w. topology
    traj = pt.iterload(trj_path, top_path)
    # load ref. structure if path is given
    ref = _load_structure(ref_str)
    # run autoimage to cluster and centre traj.
    traj = traj.autoimage()
    # align the traj. using backbone atoms
    traj = pt.align(traj, mask=aln_mask, ref=ref)
    # calculate rmsd
    data = pt.analysis.rmsd.rmsd_nofit(traj, mask=rmsd_mask, ref=ref)
    return data


def measure_rmsf(trj_path, top_path, ref_str, rmsf_mask, aln_mask='@CA,C,N,O'):
    # load the trajectory w. topology
    traj = pt.iterload(trj_path, top_path)
    # load ref. structure if path is given
    ref = _load_structure(ref_str)
    # run autoimage to cluster and centre traj.
    traj = traj.autoimage()
    # align the traj. using backbone atoms
    traj = pt.align(traj, mask=aln_mask, ref=ref)
    # calculate rmsd
    data = pt.rmsf(traj, mask=rmsf_mask, options='byres')
    return data


def extract_frame(trj_path, top, out_path,
                  ref_str=None, split=False, strip_mask=None, frame='final'):
    full_trj = pt.iterload(trj_path, top)
    full_trj = full_trj.autoimage()

   #full_trj = pt.align(full_trj, mask=':5-360@CA,C,N,O', ref=ref)
    print(f'Loaded trajectory: {trj_path}')
    N = int(full_trj.n_frames)-1 if frame=='final' else int(frame)-1
    print(f'Writing {N+1}th frame as pbd')
    pt.write_traj(out_path, full_trj, frame_indices=[N], overwrite=True)
    print(f'Saved new trajectory: {out_path}')


def measure_distance(trj_path, top_path, atom_pair, ref_str,
                     aln_mask='@CA,C,N,O'):
    # load the trajectory w. topology
    traj = pt.iterload(trj_path, top_path)
    # load ref. structure if path is given
    ref = _load_structure(ref_str)
    # run autoimage to cluster and centre traj.
    traj = traj.autoimage()
    # align the traj. using backbone atoms
    traj = pt.align(traj, mask=aln_mask, ref=ref)
    # calculate distance (A) between two pairs/groups
    data = pt.distance(traj, f"{atom_pair[0]} {atom_pair[1]}")
    return data


def measure_angle(trj_path, top_path, angle_atoms, ref_str,
                  aln_mask='@CA,C,N,O'):
    # load the trajectory w. topology
    traj = pt.iterload(trj_path, top_path)
    # load ref. structure if path is given
    ref = _load_structure(ref_str)
    # run autoimage to cluster and centre traj.
    traj = traj.autoimage()
    # align the traj. using backbone atoms
    traj = pt.align(traj, mask=aln_mask, ref=ref)
    # calculate angle between set of atoms
    data = pt.dihedral(traj=traj, mask=angle_atoms)
    return data

