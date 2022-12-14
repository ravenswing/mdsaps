"""
===============================================================================
                                ANALYSIS TOOLS
===============================================================================

    - MDAnalysis based tools for analysing trajectories
"""

import MDAnalysis as mda
from MDAnalysis.analysis import rms

import pandas as pd

from glob import glob


def _init_universe(in_str):
    # .pdb only req. 1 input
    if isinstance(in_str, str) and '.pdb' in in_str:
        return mda.Universe(in_str)
    # .r and same named .top
    elif isinstance(in_str, str):
        return mda.Universe(in_str.split('.')[0]+'.top', in_str)
    # Explicitly specified .top(FIRST) and .xtc/trr(SECOND)
    elif isinstance(in_str, list) and len(in_str) == 2:
        return mda.Universe(in_str[0], in_str[1])
    # For references, can just be int i.e. frame no.
    elif isinstance(in_str, int):
        return in_str
    # If not any of the above raise an error
    else:
        raise ValueError("Structure not recognised")


def measure_rmsd(top_path, trj_path, ref_str, rmsd_groups,
                 aln_group='backbone'):
    # Load the topology and trajectory
    traj = _init_universe([top_path, trj_path])
    # Load ref. structure if path is given
    ref = _init_universe(ref_str)
    R = rms.RMSD(traj,  # universe to align
                 ref,  # reference universe or atomgroup
                 select=aln_group,  # group to superimpose and calculate RMSD
                 groupselections=rmsd_groups,  # groups for RMSD
                 ref_frame=0).run()  # frame index of the reference
    return R


'''
def calculate_rmsd(wd, trj_rgx, top_rgx, ref_rgx,
                   replicas=False, out_path=None):

    out = f"{wd}/rmsd.h5" if out_path is None else out_path
    for trj in glob(trj_rgx, recursive=True):
        top_lst = glob('/'.join(trj.split('/')[:-1]) + '/' + top_rgx)
        assert len(top_lst) == 1, "TOP Rgx not unique"
        ref_lst = glob('/'.join(trj.split('/')[:-1]) + '/' + ref_rgx)
        assert len(ref_lst)==1, 'REF Rgx not unique'

        data = measure_rmsd(top_lst[0],
                            trj,
                            ref_lst[0],
                            rmsd_groups='backbone')
    # Output as a dataframe in a compressed hdf5 file
    if '.h5' in out:
        df = pd.DataFrame(columns=[''])

    # Output as pickled numpy arrary
    elif '.p' in out:
        print("PICKLE")
    # Unknown ouput type returns error. 
    else:
        raise ValueError('Output type not recognised')








        out_file = f"{DATA_DIR}/{system}+{lig}/06-MetaD/ligand_rmsd.p"
        with open(out_file, 'wb') as f:
            pickle.dump(lig_rmsd, f)
'''
