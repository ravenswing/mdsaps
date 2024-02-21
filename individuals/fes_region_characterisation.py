from __future__ import print_function
from functools import partial
import mdtraj as md
from multiprocessing import Pool
import numpy as np
import pandas as pd
from time import time
import sys

sys.path.append('/home/rhys/phd_tools/SAPS')
from load import colvar


def import_basins(csv: str):
    return {k: b.iloc[:, 2:].values.tolist()
            for k, b in pd.read_csv(csv).groupby('sys')}


def calc_rmsd(frame, trajectory, atoms):
    return md.rmsd(trajectory, trajectory, frame, atom_indices=atoms)


def calc_3D_dist(x1, y1, z1, x2, y2, z2):
    return np.sqrt(np.square(x2-x1) + np.square(y2-y1) + np.square(z2-z1))


def com_dist(traj, atom_groupA, atom_groupB):
    # Split the trajectores to just portal & ligand atoms
    trajA = traj.atom_slice(atom_groupA)
    trajB = traj.atom_slice(atom_groupB)
    # Compute coordinates of both centers of mass, store in DataFrames
    crdsA = pd.DataFrame(md.compute_center_of_mass(trajA),
                         columns=['xa', 'ya', 'za'])
    crdsB = pd.DataFrame(md.compute_center_of_mass(trajB),
                         columns=['xb', 'yb', 'zb'])
    # Merge DataFrames
    crds = pd.concat([crdsA, crdsB], axis=1)
    # Calculate 3D distance between the two sets of coords.
    crds['dist'] = list(map(calc_3D_dist,
                            crds.xa, crds.ya, crds.za,
                            crds.xb, crds.yb, crds.zb))
    # Convert distances to Angstroms
    crds.dist = crds.dist.multiply(10)
    # Return distances as Series
    return crds.dist


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
    max_s = np.exp(-beta*distances / distances.std()).sum(axis=1).max()
    # print(index)
    # Extract centroid structure as pdb
    centroid = traj[index]
    # print(centroid)
    centroid.save_pdb(out_name)
    return [index, max_s, distances.mean(), distances.std()]


if __name__ == "__main__":
    """
    # Define all the systems to work with
    METHODS = ['fun-metaD', 'fun-RMSD']
    #SYSTS = ['a2b1', 'a2b2']
    #REPS = ['R'+str(x) for x in np.arange(2)+1]
    """
    METHODS = ['fun-metaD']
    SYSTS = ['a2b1', 'a2b2']
    LIGS = ['A769', 'PF739', 'SC4', 'MT47', 'MK87']
    REPS = ['R1', 'R2', 'R3', 'R4']

    # Define the source and output directories
    DATA_DIR = '/home/rhys/Storage/ampk_metad_all_data'
    OUT_DIR = '/home/rhys/Dropbox/RESEARCH/AA_RHYS/BB_AMPK/Plots/re-entry'

    portals = {'a2b1-front': [19, 41, 81, 278, 304, 305, 306],
               'a2b1-back': [4, 10, 13, 14, 301, 302, 303],
               'a2b2-front': [20, 41, 81, 278, 280, 281, 306, 307],
               'a2b2-back': [4, 12, 13, 302, 303, 304]}
    lig_sides = {'A769-LHS': ['C2', 'C3', 'C4', 'C9', 'C11', 'C16', 'O2'],
                 'A769-RHS': ['C1', 'C10', 'C14', 'C15', 'C17', 'C18', 'C19',
                              'C20', 'O1', 'O3', 'N1', 'N2', 'S1'],
                 'MT47-LHS': ['C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'O1'],
                 'MT47-RHS': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8',
                              'C9', 'N1', 'O2', 'O3', 'C22', 'N2'],
                 'MK87-LHS': ['C1', 'C2', 'C3', 'C4', 'C10', 'C15'],
                 'MK87-RHS': ['C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'O2',
                              'O3', 'O4'],
                 'PF739-LHS': ['C20', 'C21', 'C22', 'C23', 'O5'],
                 'PF739-RHS': ['C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'O2',
                               'O3', 'O4'],
                 'SC4-LHS': ['C2', 'C3', 'C4', 'C6', 'C16', 'C23', 'O3'],
                 'SC4-RHS': ['C1', 'C5', 'C7', 'C18', 'C12', 'C22', 'C14',
                             'O1', 'O2']}
    for method in METHODS:
        for system in SYSTS:
            with open(f"{OUT_DIR}/{method}_{system}_dists.csv", 'w') as f:
                f.write(('system,ligand,replica,door,total_frames,'
                        'lhs,rhs,similar\n'))
            lig_residue = 369 if system == 'a2b1' else 368
            for lig in LIGS:
                for rep in REPS:
                    wd = f"{DATA_DIR}/{method}/{system}+{lig}/{rep}"
                    # Load the trajectory
                    traj = md.load(f'{wd}/{system}+{lig}_{rep}_GISMO.xtc',
                                   top=f'{wd}/md_dry.pdb')
                    # Align the trajectory - to first frame
                    traj_aligned = traj.superpose(traj, 0)
                    # Load in the COLVAR file
                    clv = colvar(f'{wd}/{system}+{lig}_{rep}_GISMO.colvar').drop([1]).reset_index(drop=True)
                    n_frames = len(clv.index)
                    print(n_frames)

                    for door in ['front', 'back']:
                        b = [0.8, 1.5, 0.0, 1.0] if door == 'front' else [0.0, 1.0, 0.75, 1.8]
                        # Extract indices from COLVAR
                        indices = clv.loc[(clv['pp.proj'].between(b[0], b[1]))
                                          & (clv['pp.ext'].between(b[2], b[3]))].index.values
                        print(len(indices))
                        print(f"Number of structures in {door} basin: {len(indices)}")
                        # Slice the trajectory to get frames from each basin
                        b_frames = traj_aligned.slice(indices)
                        # Group A = portal
                        portal_res = portals[f"{system}-{door}"]
                        #print(f"(resid {' or resid '.join([str(x) for x in portal_res])}) and rescode != H")
                        portal_atoms = traj.topology.select(f"(resid {' or resid '.join([str(x) for x in portal_res])}) and rescode != H")

                        dists = {}
                        for side in ['LHS', 'RHS']:
                            ligand_names = lig_sides[f"{lig}-{side}"]
                            ligand_atoms = traj.topology.select(f"residue {lig_residue} and (name {' or name '.join(ligand_names)} )")
                            #print(f"resid {lig_residue} and (name {' or name '.join(ligand_names)} )")
                            dists[side] = com_dist(b_frames, portal_atoms, ligand_atoms)
                        df = pd.DataFrame.from_dict(dists)
                        # Filter those with minimum distance less than 10 A
                        df = df.loc[df.abs().min(axis=1) <= 10]
                        # Calculate the difference between the two sides
                        df['dif'] = df.LHS - df.RHS
                        # Total number of frames in region
                        tot = len(df.index)
                        # Number of frames where difference is small (< 1 A)
                        sim = len(df[df.dif.abs() <= 1.0].index)
                        # Number of frames where ligand LHS is closer
                        lhs = len(df[(df.dif.abs() > 1.0) & (df.dif < 0)].index)
                        # Number of frames where ligand RHS is closer
                        rhs = len(df[(df.dif.abs() > 1.0) & (df.dif > 0)].index)
                        # Write line to csv file with all data (as decimals)
                        with open(f"{OUT_DIR}/{method}_{system}_dists.csv", 'a') as f:
                            f.write((f"{system},{lig},{rep},{door},"
                                     f"{tot/n_frames:.5f},{lhs/tot:.5f},{rhs/tot:.5f},"
                                     f"{sim/tot:.5f}"
                                     "\n"))
'''
    for system in SYSTS:
        for lig in LIGS:
            #subprocess.run(f"mkdir -p {DATA_DIR}//{lig}/", shell=True)
            with open(f"{DATA_DIR}/{system}_{lig}/cluster_stats.csv", 'w') as f:
                f.write(('system,lig,basin,n_frames,centroid_index,max_simil,'
                         'matrix_rmsd_avg,martix_rmsd_std'))
            for rep in REPS:
                wd = f"{DATA_DIR}/{system}_{lig}/{rep}"
                # Load the trajectory
                traj = md.load(f'{wd}/{system}_{lig}_{rep}_GISMO.xtc',
                               top=f'{wd}/md_dry.pdb')
                # Slice heavy atoms
                atom_ids = [a.index for a in traj.topology.atoms
                            if a.element.symbol != 'H']
                # Load in the COLVAR file
                clv = colvar(f'{wd}/{system}_{lig}_{rep}_GISMO.colvar').drop([1]).reset_index(drop=True)
                # Loop over all basins for current system
                for i, b in enumerate(basins[f"{system}_{lig}"]):
                    # Extract indices from COLVAR
                    indices = clv.loc[(clv['pp.proj'].between(b[0], b[1]))
                                      & (clv['pp.ext'].between(b[2], b[3]))].index.values
                    print(f"Number of structures in basin {i}: {len(indices)}")
                    # Slice the trajectory to get frames from each basin
                    b_frames = traj.slice(indices)
                    # Prepare partial func. for paralleisation
                    pf = partial(calc_rmsd,
                                 trajectory=b_frames,
                                 atoms=atom_ids)
                    # Cluster and save centroid using MDTraj
                    d = save_centroid(b_frames, pf,
                                      (f"{wd}/"
                                       f"{system}_{lig}_{rep}_b{i}.pdb"))
                    d = [system, lig, i, len(indices)] + d
                    with open(f"{DATA_DIR}/{system}_{lig}/cluster_stats.csv", 'a') as f:
                        f.write('\n'+','.join([str(x) for x in d]))
'''
