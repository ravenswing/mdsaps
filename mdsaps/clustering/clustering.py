import numpy as np
import os
import MDAnalysis as mda
from MDAnalysis.analysis import encore
from MDAnalysis.analysis.encore.clustering import ClusteringMethod as clm
from pathlib import Path

from .. import tools, plot, load
from ..config import CORES


def get_indices(
    colvar_path: str, cv_bounds, colvar_stride: int = None, return_info: bool = False
):
    colvar = load.colvar(colvar_path)
    info = {"Initial Colvar Frames": len(colvar.index)}
    if colvar_stride:
        colvar = colvar.iloc[::colvar_stride, :].reset_index(drop=True)
    info["Colvar Frames w. Stride"] = len(colvar.index)

    # while colvar.time.iloc[1] != initial.trajectory[1].time:
    # colvar = colvar.drop(1).reset_index(drop=True)

    for cv, bounds in cv_bounds.items():
        colvar = colvar.loc[colvar[cv].between(bounds[0], bounds[1])]
    indices = colvar.index.values
    info["Percentage"] = (len(indices) / info["Colvar Frames w. Stride"]) * 100

    return indices if not return_info else indices, info


def make_selective_traj(
    traj_path: str, top_path: str, out_path: str, ref_path: str, indices
) -> None:
    # Create path for temporary selected trajectory.
    tmp_file = Path("/tmp/unaligned").with_suffix(Path(traj_path).suffix)
    # Produce a trajectory with only the frames that you want.
    tools.slice_traj(
        traj_path,
        top_path,
        str(tmp_file),
        indices,
        _check_frames=True,
    )
    # Align the trajectory to a reference structure.
    tools.align_traj(
        str(tmp_file),
        top_path,
        ref_path,
        out_path,
    )
    # Delete the temporary file.
    tmp_file.unlink()


def unpack_collection(cluster_collection):
    # ClusterCollections are not sorted nor accessible via index
    # BUT they can be iterated:
    clusters = [c for c in cluster_collection]

    # Find the correct order of the clusters based on their sizes.
    ordered_sizes = zip(
        cluster_collection.get_ids(), [a.size for a in cluster_collection]
    )
    ordered_sizes = sorted(ordered_sizes, key=lambda x: x[1], reverse=True)
    new_order, cluster_sizes = zip(*ordered_sizes)

    # Reorder the clusters to be in size order
    clusters[:] = [clusters[i] for i in new_order]

    # Get the total number of frames that were clustered.
    n_frames = sum(cluster_sizes)

    # Calculate the percentage of frames that each cluster represents.
    percentages = [(s / n_frames) * 100 for s in cluster_sizes]

    return clusters, cluster_sizes, percentages, n_frames


def plot_sizes(cluster_collection, out_path: str) -> None:
    # Extract ordered information from ClusterCollection
    _, sizes, percentages, _ = unpack_collection(cluster_collection)

    plot.cluster_sizes(sizes, percentages, out_path)


def save_centroids(
    cluster_collection,
    u,
    out_dir: str,
    out_name: str = "cluster",
    pdbs: bool = True,
    out_perc: bool = True,
    timestamp_csv: bool = True,
    _warn=True,
) -> None:
    # Extract ordered information from ClusterCollection
    clusters, sizes, percentages, _ = unpack_collection(cluster_collection)

    # Save centroid as pdb, with ID and percentage labels
    if pdbs:
        for i, size in enumerate(sizes):
            pdb_name = (
                f"{out_dir}/{out_name}{i}_{percentages[i]:.0f}%.pdb"
                if out_perc
                else f"{out_dir}/{out_name}.pdb"
            )
            u.trajectory[clusters[i].centroid]
            with mda.Writer(pdb_name, u.atoms.n_atoms) as W:
                W.write(u.atoms)

    if timestamp_csv:
        if _warn:
            print(
                "!!! WARNING: MDA Clustering removes original timestamp information !!!"
            )
            print(
                "    --> Solution Re-initialise initial universe in order to get a\n        correct timestamp csv."
            )
        lines = ["Cluster No.,Centroid ID,Initial Trajectory Timestamp\n"]
        for i, c in enumerate(clusters):
            lines.append(f"{i},{c.centroid},{u.trajectory[c.centroid].time}\n")
        with open(f"{out_dir}/centroid_timestamps.csv", "w") as f:
            f.writelines(lines)


def kmeans(u, n_clusters: int, cluster_selection: str = "backbone"):
    # cluster_selection = 'protein or resname S2P'  # full protein
    # cluster_selection = 'resname MOL'  # ligand

    # Set up MDA ClusteringMethod for kmeans
    kmeans = clm.KMeans(
        n_clusters,  # no. of clusters
        init="k-means++",  # default
        algorithm="auto",
    )  # default

    # Run the clustering, using config default number of cores
    cluster_collection = encore.cluster(
        u, select=cluster_selection, method=kmeans, ncores=CORES
    )
    # Gives a MDA ClusterCollection as output
    return cluster_collection


def kmeans_scan(
    traj_path: str,
    top_path: str,
    out_dir: str,
    cluster_selection: str = "backbone",
    n_min: int = 2,
    n_max: int = 6,
) -> None:
    # Create MDA universe from trajectory.
    u = tools._init_universe([top_path, traj_path])

    # For each number of clusters...
    for n in np.arange(n_min, n_max + 1):
        # ...make the output directory
        dir = f"{out_dir}/N={n}"
        os.makedirs(dir, exist_ok=True)
        # ...perform the kmeans clustering to make ClusterCollection.
        cluster_collection = kmeans(u, n, cluster_selection)
        # running the clustering affects the universe e.g. removes original time information
        # re-initialise to preserve and save
        u = tools._init_universe([top_path, traj_path])
        # ...save the centroids as pdbs.
        save_centroids(
            cluster_collection, u, dir, out_name=f"n={n}_cluster", _warn=False
        )
        # ...plot the sizes of the clusters.
        plot_sizes(cluster_collection, f"{dir}/Cluster_Sizes_n={n}.png")


def single_centroid(
    traj_path: str,
    top_path: str,
    out_path: str,
    cluster_selection: str = "backbone",
) -> None:
    top_path = (
        str(Path(traj_path).parent / Path(top_path))
        if "/" not in top_path
        else top_path
    )
    out_path = (
        Path(traj_path).parent / Path(out_path)
        if "/" not in out_path
        else Path(out_path)
    )
    # Create MDA universe from trajectory.
    u = tools._init_universe([top_path, traj_path])
    # ...perform the kmeans clustering to make ClusterCollection.
    cluster_collection = kmeans(u, 1, cluster_selection)
    # running the clustering affects the universe e.g. removes original time information
    # re-initialise to preserve and save
    u = tools._init_universe([top_path, traj_path])
    # ...save the centroids as pdbs.
    save_centroids(
        cluster_collection,
        u,
        out_path.parent,
        out_name=out_path.stem,
        out_perc=False,
        _warn=False,
    )


def convert_timestamps(
    cluster_collection,
):
    print("AAAAAAAAAAAAAAAAAAAAAAAAA")
