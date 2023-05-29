from __future__ import print_function
from functools import partial
import mdtraj as md
from multiprocessing import Pool
import numpy as np
import pandas as pd
from time import time
import sys

sys.path.append('/home/rhys/phd_tools/python_scripts/')
from load_data import colvar


def import_basins(csv):
    return {k: b.iloc[:, 2:].values.tolist()
            for k, b in pd.read_csv(csv).groupby('sys')}


def calc_rmsd(frame, trajectory, atoms):
    return md.rmsd(trajectory, trajectory, frame, atom_indices=atoms)


def save_centroid(traj, per_frame, out_name):
    # Prepare frames for paralleisation
    frames = range(traj.n_frames)
    # Time the start of the process
    t0 = time()
    # Start pool of workers (multiprocessing) w. 8 cores
    with Pool(8) as pool:
        out = pool.map(per_frame, frames)
    # Reshape output
    distances = np.asarray(out).T
    # Output time taken
    t1 = time()
    print(f"  Time elapsed: {t1-t0:.2f} s")

    # Identify index of centroid frame from trajectory
    beta = 1  # I DO NOT KNOW WHAT BETA IS OR DOES!
    index = np.exp(-beta*distances / distances.std()).sum(axis=1).argmax()
    # print(index)
    # Extract centroid structure as pdb
    centroid = traj[index]
    # print(centroid)
    centroid.save_pdb(out_name)


if __name__ == "__main__":

    # Define all the systems to work with
    # SYSTS = ['a2b1', 'a2b2']
    # LIGS = ['A769', 'PF739', 'SC4', 'MT47', 'MK87']
    # REPS = ['R'+str(x) for x in np.arange(2)+1]

    SYSTS = ['a2b1']
    LIGS = ['A769']
    REPS = ['R2']
    # Define the source and output directories
    DATA_DIR = '/media/rhys/Storage/ampk_metad_all_data'
    OUT_DIR = '/home/rhys/Clustering/'

    # Extract basin information from .csv file
    basins = import_basins(f"{OUT_DIR}/basins.csv")

    for system in SYSTS:
        for lig in LIGS:
            for rep in REPS:
                wd = f"{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}"
                # Load the trajectory
                traj = md.load(f'{wd}/{system}+{lig}_{rep}_GISMO.xtc',
                               top=f'{wd}/md_dry.pdb')
                # Slice heavy atoms
                atom_ids = [a.index for a in traj.topology.atoms
                            if a.element.symbol != 'H']
                # Prepare partial func. for paralleisation
                pf = partial(calc_rmsd,
                             trajectory=traj,
                             atoms=atom_ids)
                # Load in the COLVAR file
                clv = colvar(f'{wd}/{system}+{lig}_{rep}_GISMO.colvar').drop([1]).reset_index(drop=True)
                # Loop over all basins for current system
                for i, b in enumerate(basins[f"{system}+{lig}"]):
                    # Extract indices from COLVAR
                    indices = clv.loc[(clv['pp.proj'].between(b[0], b[1]))
                                      & (clv['pp.ext'].between(b[2], b[3]))].index.values
                    print(f"Number of structures in basin {i}: {len(indices)}")
                    # Slice the trajectory to get frames from each basin
                    b_frames = traj.slice(indices)
                    # Cluster and save centroid using MDTraj
                    save_centroid(b_frames, pf,
                                  (f"{OUT_DIR}/mdtraj/"
                                   f"{system}+{lig}_{rep}_b{i}.pdb"))
