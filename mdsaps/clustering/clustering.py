import matplotlib.pyplot as plt
import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.analysis import encore, align
from MDAnalysis.analysis.encore.clustering import ClusteringMethod as clm
from pathlib import Path

from .. import tools, plot, load
from ..config import *


def get_indices(colvar_path: str,  cv_bounds, colvar_stride: int = None):
    
    colvar = load.colvar(colvar_path)
    if colvar_stride:
        colvar = colvar.iloc[::colvar_stride, :].reset_index(drop=True)
    print(f"Colvar Frames: {len(colvar.index)}")

    # while colvar.time.iloc[1] != initial.trajectory[1].time:
    # colvar = colvar.drop(1).reset_index(drop=True)

    for cv, bounds in cv_bounds.items():
        colvar = colvar.loc[colvar[cv].between(bounds[0], bounds[1])]
    indices = colvar.index.values

    return indices


def selective_traj(traj_path: str, top_path: str, out_path: str, indices) -> None:

    top_path = str(Path(traj_path).parent / Path(top_path)) if '/' not in top_path else top_path
    out_path = str(Path(traj_path).parent / Path(out_path)) if '/' not in out_path else out_path

    initial = tools._init_universe([top_path, traj_path])
    print(f"Trajectory Frames: {initial.trajectory.n_frames}")
    
    with mda.Writer(out_path, initial.atoms.n_atoms) as W:
        for idx in indices:
            initial.trajectory[idx]
            W.write(initial.atoms)


def align_traj(traj_path: str, top_path: str, ref: str, out_path: str,
               selection: str='backbone'): 
    
    top_path = str(Path(traj_path).parent / Path(top_path)) if '/' not in top_path else top_path
    ref = str(Path(traj_path).parent / Path(ref)) if '/' not in ref else ref
    out_path = str(Path(traj_path).parent / Path(out_path)) if '/' not in out_path else out_path
    
    mobile = tools._init_universe([top_path, traj_path]) 
    reference = tools._init_universe(ref)
    
    aligner = align.AlignTraj(mobile, reference, select=selection, filename=out_path).run()


def kmeans(u, n_clusters: int, cluster_selection: str = "backbone"): 
    # cluster_selection = 'protein or resname S2P'  # full protein
    # cluster_selection = 'resname MOL'  # ligand

    # n_clusters = 2

    kmeans = clm.KMeans(n_clusters,  # no. of clusters
                        init='k-means++',  # default
                        algorithm="auto")    # default

    cluster_collection = encore.cluster(u,
                                        select=cluster_selection,
                                        method=kmeans,
                                        ncores=CORES)
    return cluster_collection


def unpack_collection(cluster_collection):
    # ClusterCollections are not sorted nor accessible via index
    # BUT they can be iterated:
    clusters = [c for c in cluster_collection]
    
    # Find the correct order of the clusters based on their sizes.
    ordered_sizes = zip(cluster_collection.get_ids(), [a.size for a in cluster_collection])                    
    ordered_sizes = sorted(ordered_sizes, key=lambda x: x[1], reverse=True)
    new_order, cluster_sizes = zip(*ordered_sizes) 
    
    # Reorder the clusters to be in size order
    clusters[:] = [clusters[i] for i in new_order]
   
    # Get the total number of frames that were clustered.
    n_frames = sum(cluster_sizes)

    # Calculate the percentage of frames that each cluster represents.
    percentages = [(s/n_frames)*100 for s in cluster_sizes]
    
    return clusters, cluster_sizes, percentages, n_frames


def save_centroids(cluster_collection, u, out_dir: str, 
                   out_name='Cluster'): 

    clusters, sizes, percentages, n_frames = unpack_collection(cluster_collection)

    for i, size in enumerate(sizes): 
        u.trajectory[clusters[i].centroid]
        with mda.Writer(f'{out_dir}/{out_name}{i}_{percentages[i]:.0f}%.pdb', u.atoms.n_atoms) as W:
            W.write(u.atoms)


def plot_sizes(cluster_collection, out_path):

    _, sizes, percentages, _ = unpack_collection(cluster_collection) 

    fig, ax = plt.subplots(1,1, figsize=(10,6))

    x_axis = np.arange(len(sizes))
    bar = ax.bar(x_axis, sizes, color='#089682')
    plt.bar_label(bar, labels=[f"{x:.1f}%" for x in percentages])

    ax.set_xlabel('Cluster No.')
    ax.set_xticks(x_axis)
    ax.set_ylabel('Cluster Sizes')
    ax.set_ylim(0, sizes[0]+.2*sizes[0])
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()


def kmeans_scan(traj_path, top_path, selection, out_dir,
                n_min: int = 2, n_max: int = 6):

    u = tools._init_universe([top_path, traj_path])

    for n in np.arange(n_min, n_max):
        cluster_collection = kmeans(u, n, selection)
