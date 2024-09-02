"""
===============================================================================
                    Gromacs Organisation n' Analysis Tools
===============================================================================

    - sumhills
"""

import logging
import pandas as pd
import subprocess
from pathlib import Path
from glob import glob
from typing import Optional

from . import load
from .config import GMX

log = logging.getLogger(__name__)
log.info("G.O.A.T. (Gromacs Organisation n' Analysis Tools) Loaded")


def run_sumhills(
    wd: str,
    out_name: str,
    new_dir: str = "fes",
    name: str = "HILLS",
    stride: Optional[float] = None,
    cv: Optional[str] = None,
):
    """Outputs:
    - FES
    - FES over time (with stride)
    - 1D FES (with cv)
    """
    hills_file = f"{wd}/{name}"
    log.info(f"Running Sum_hills for {hills_file}")

    # Create FESs over time is stride is provided
    if stride is not None:
        # TODO -> Make dirs
        # Make a new directory to hold output
        subprocess.run(f"mkdir -p {wd}/{new_dir}", shell=True, check=True)
        # Adjust output name for new directory
        out_name = f"{new_dir}/{out_name}"
        # Add flag for plumed command
        st_flags = ["--stride", f"{stride}"]
    else:
        st_flags = []
        out_name = f"{out_name}_FES"

    # Create 1D FES if cv is specified, add flag for plumed command (300K!)
    cv_flags = ["--idw", f"{cv}", "--kt", "2.49"] if cv is not None else []

    # Construct plumed command
    cmd = (
        [
            "plumed",
            "sum_hills",
            "--hills",
            hills_file,
            "--outfile",
            f"{wd}/{out_name}",
            "--mintozero",
        ]
        + st_flags
        + cv_flags
    )
    log.debug(f"{' '.join(cmd)}")

    # Execute the plumed sum_hills command
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
        )


def run_reweight(
    wd: str,
    cvs: list[str],
    fes_prefix: str = "FES",
    colvar_name: str = "COLVAR",
    out_name: str = "FES_REW",
    bias_factor: float = 10.0,
):
    # TODO Add cv = None & 2 CVs in FES -> do 2D reweight

    # SCRIPT = "/home/rhys/mdsaps/mdsaps/lib/auxiliary_scripts/reweight.py"
    SCRIPT = "/home/rhys/mdsaps/mdsaps/lib/auxiliary_scripts/wip_rew.py"

    colvar_path = f"{wd}/{colvar_name}"
    colvar = load.colvar(colvar_path)
    print(colvar)
    print(colvar.columns)
    bias_column = colvar.columns.get_loc("meta.bias") + 1
    n_fes = len(glob(f"{wd}/fes/{fes_prefix}*.dat"))
    if isinstance(cvs, str):
        column = colvar.columns.get_loc(cvs) + 1
        fes_column = 2
    elif isinstance(cvs, list) and len(cvs) == 1:
        column = colvar.columns.get_loc(cvs[0]) + 1
        fes_column = 2
    elif isinstance(cvs, list) and len(cvs) == 2:
        column = " ".join([str(colvar.columns.get_loc(cv) + 1) for cv in cvs])
        fes_column = 3

    else:
        print("ERROR: Please input CVs for 1D or 2D FES.")

    log.info(
        f"Reweighting FES - Using COLVAR column(s) {column} for cvs: {' '.join(cvs)}"
    )

    command = [
        "python",
        SCRIPT,  # path to python script
        f"-bsf {bias_factor}",  # BIASFACTOR used in simulation
        f"-fpref {wd}/fes/{fes_prefix}",  # prefix for FESs
        f"-nf {n_fes}",  # number of FESs
        f"-fcol {fes_column}",  # column of free energy in FES
        f"-colvar {colvar_path}",  # (default value)
        f"-biascol {bias_column}",  # column in COLVAR containing energy bias
        f"-rewcol {column}",  # column(s) to reweight over
        f"-outfile {out_name}",
        "-v",
    ]
    log.debug(" ".join(command))
    print(" ".join(command))
    try:
        subprocess.run(" ".join(command), shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:",
            error.returncode,
            ". Output:",
            error.output.decode("utf-8"),
        )


def sumhills_convergence(
    wd: str,
    out_name: str,
    every: int = 50,
    name: str = "HILLS",
    cv: Optional[str] = None,
    new_dir: str = "convergence",
    overwrite: bool = True,
):
    new_dir_path = f"{wd}/{new_dir}"

    if overwrite and Path(new_dir_path).exists():
        print("removing")
        subprocess.run(f"rm {new_dir_path}/*.dat", shell=True)

    run_sumhills(
        wd,
        out_name=f"{out_name}_",
        new_dir=new_dir,
        stride=every / 0.002,
        name=name,
        cv=cv,
    )

    for fes in Path(new_dir_path).glob("*.dat"):
        fes_number = int(str(fes.stem).split("_")[-1])
        new_path = fes.with_stem(
            f"{'_'.join(str(fes.stem).split('_')[:-1])}_{(fes_number) * every}"
        )
        fes.replace(new_path)


"""
def run_trjconv(options: list[str], echo: list[str] = None):

    out_file = ["#!/bin/bash", "touch ./COMPLETED"]

    if echo:
        command = f"echo {' '.join(echo)} | {GMX} trjconv {' '.join(options)}"
    else:
        command = f"{GMX} trjconv {' '.join(options)}"

    out_file.insert(1, command)
    with open('/tmp/trjconv_in.sh', 'w') as f:
        f.writelines(out_file)

    subprocess.run('bash /tmp/trjconv_in.sh', shell=True)

    signal_path = Path("/tmp/COMPLETED")
    while not signal_path.exists():
        sleep(5)

    # remove the file
    signal_path.unlink()

    # return "exitcode"
    return signal_path.exists():
"""


def cut_traj(
    trj_path: str,
    tpr: str,
    out_path: str,
    dt: int = 100,
    ndx: str = "i.ndx",
    apo: bool = False,
) -> None:
    """cutdown the trajectories using Gromacs trjconv ready for GISMO"""
    # Assume working directory is same as traj if not specified
    tpr = tpr if "/" in tpr else "/".join(trj_path.split("/")[:-1]) + "/" + tpr
    out_path = (
        out_path
        if "/" in out_path
        else "/".join(trj_path.split("/")[:-1]) + "/" + out_path
    )
    ndx = ndx if "/" in ndx else "/".join(trj_path.split("/")[:-1]) + "/" + ndx
    log.info(f"Cutting Trajectory {trj_path.split('/')[-1]}")
    out_group = "Protein" if apo else "Protein_LIG"
    # Create the trjconv command from user input
    cmd = [
        f"echo Backbone {out_group} |",
        GMX,
        "trjconv ",
        f"-s {tpr}",
        f"-f {trj_path}",
        f"-o {out_path}",
        f"-n {ndx}",
        f"-dt {str(dt)}",
        "-fit rot+trans",
    ]
    log.debug(f"{' '.join(cmd)}")
    # Run the trjconv command
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
        )


def gismo_colvar(wd: str, in_colvar: str = "COLVAR", out_colvar: str = "GISMO.colvar"):
    """combine old and reweighted colvars"""
    # Load in the original COLVAR
    old_col = load.colvar(f"{wd}/{in_colvar}", "as_pandas")

    # Cutdown old COLVAR to match trajectories by selecting every 5th line
    old_col = old_col.iloc[::5, :]
    # Add every 10th line (and the second line) for GISMO colvar
    gis_col = old_col.iloc[:2, :]
    gis_col = pd.concat([gis_col, old_col.iloc[10::10, :]], ignore_index=True)

    # Define path for the output GISMO COLVAR file
    gismo_col_path = f"{wd}/{out_colvar}"
    # Add the header line to this new COLVAR
    with open(gismo_col_path, "w") as f:
        f.write("#! FIELDS " + " ".join(list(gis_col.columns.values)) + "\n")
    # Save the cutdown GISMO COLVAR
    gis_col.to_csv(gismo_col_path, sep=" ", header=False, index=False, mode="a")
    print(f"Successfully converted {in_colvar} to {out_colvar}.")


def _bound_check(x, bound, unbound):
    """Establish bound, unbound or in middle"""
    # Check bounds are correct
    assert len(bound) == 2, "Bound requires 2 inputs"
    # Calculate upper bound limit (avg. + std. dev. for proj)
    threshold = bound[0] + bound[1]
    # Value of 1 = bound
    if x < threshold:
        return 1
    # Value of 2 = un-bound
    elif x > unbound:
        return 2
    # Value of 0 = in the middle
    else:
        return 0


def _identify_recross(data, metric, bound, unbound):
    """Count the number of recrossings"""
    # Calculate status of ligand position: 1 = bound, 2 = unbound
    data["status"] = data[metric].apply(_bound_check, args=([bound, unbound]))
    # Remove data without a status i.e. not bound or unbound
    middle_ind = data[data.status == 0].index
    data.drop(middle_ind, inplace=True)
    # Calculate differences in status column (diff. of 1 = transition)
    data["diffs"] = data.status.diff()
    # Identify transitions
    rx_ind = data[data.diffs != 0].index
    # Extract times as list for plotting
    rx = [1.0] + [t for t in data.loc[rx_ind[1:]].t.tolist()][1::2]
    # Count number of recrossings
    N = int((len(rx) - 1))
    # Output number of RX and list of RX times
    return N, rx


def rx(
    dir_dict,
    var,
    bound=None,
    unbound=None,
    from_colvar=None,
    from_hdf=None,
    outpath="rx",
    columns=["system", "lig", "rep"],
):
    # Create storage df
    data = pd.DataFrame(columns=columns + ["number", "rx"])

    for wd, ids in dir_dict.items():
        # Check that the ids and output columns match shape
        assert len(ids) == len(columns), "IDs given do not match columns."
        # Check if both data sources are given
        if from_colvar is not None and from_hdf is not None:
            raise Exception("from_hdf and from_colvar are mutually exclusive")
        # Load a format colvar data
        elif from_colvar is not None:
            # Load the colvar into DataFrame
            df = load.colvar(f"{wd}/{from_colvar}", "as_pandas")
            # Rename time column
            df.rename(columns={"time": "t"}, inplace=True)
            # Remove unnecessary columns
            df.drop(columns=[col for col in df if col not in ["t", var]], inplace=True)
            # Convert CVs to Angstroms
            df[var] = df[var].multiply(10)
        # Load and format stored HDF data
        elif from_hdf is not None:
            # Load hdf file
            df = pd.read_hdf(from_hdf, key="df")
            # Collapse multi-indexing
            df.columns = df.columns.map("-".join)
            # Column name to match, based on ids
            var = "-".join(ids)
            # Create time column from df index
            df["t"] = df.index.to_series()
            # Remove unnecessary columns
            df = df.drop(columns=[col for col in df if col not in ["t", var]])
        # Check if any source provided
        else:
            raise Exception("no source for data")
        # Convert time to ns
        df.t = df.t.multiply(0.001)
        # Identify bound state from initial CV value if none given.
        if bound is None:
            bnd = [df[var].iloc[0], 1.0]
        # If only int is given assume +/- = 0
        elif isinstance(bound, float):
            bnd = [bound, 0.0]
        elif isinstance(bound, list):
            bnd = bound
        # Check if bound is either a list or an int
        else:
            raise Exception("Bound must be float or list")
        # Identify unbound state from max CV value - 10 if none given.
        unb = df[var].max() - 10 if unbound is None else unbound
        # Run recrossing counting function
        N, rx = _identify_recross(df, var, bound=bnd, unbound=unb)
        # Add values to storage dataframe
        data.loc[len(data.index)] = ids + [N, rx]
    # Save data
    data.to_hdf(f"{outpath}.h5", key="df", mode="w")


def calculate_delta_g(fes_path, A, B, vol_corr=0, CVs=None):
    fes_data, cvs = load.fes(fes_path)
    # Convert the CVs to Angstroms:
    fes_data[cvs[0]] = fes_data[cvs[0]].multiply(10)
    fes_data[cvs[1]] = fes_data[cvs[1]].multiply(10)
    # Isolate the values that correspond to the basins
    basin_A = fes_data[
        (fes_data[cvs[0]].between(A[0], A[1])) & (fes_data[cvs[1]].between(A[2], A[3]))
    ].free
    basin_B = fes_data[
        (fes_data[cvs[0]].between(B[0], B[1])) & (fes_data[cvs[1]].between(B[2], B[3]))
    ].free
    # Calculate the dG from the minimum value in each basin (bound - unbound)
    delta_g = basin_A.min() - basin_B.min()
    # Convert to kcal and apply volume correction for funnel
    delta_g = (delta_g / 4.184) + vol_corr
    return delta_g


def reconstruct_traj(
    trj_path: str,
    tpr: str,
    out_path: Optional[str] = None,
    ndx: str = "i.ndx",
    out_group: str = "System",
    ignore_pbc: bool = False,
):
    # Assume working directory is same as traj if not specified
    tpr = tpr if "/" in tpr else "/".join(trj_path.split("/")[:-1]) + "/" + tpr
    ndx = ndx if "/" in ndx else "/".join(trj_path.split("/")[:-1]) + "/" + ndx
    if out_path is None:
        out_path = (
            f"{'/'.join(trj_path.split('/')[:-1])}/"
            f"{trj_path.split('/')[-1][:-4]}_final.xtc"
        )
    elif "/" not in out_path:
        out_path = "/".join(trj_path.split("/")[:-1]) + "/" + out_path

    # Step 1: -pbc whole, produces tmp1.xtc
    #    N.B. does not run if ignore_pbc as the only change is the -pbc flag
    cmd = [
        f"echo {out_group} |",
        GMX,
        "trjconv",
        f"-f {trj_path}",
        f"-s {tpr}",
        f"-n {ndx}",
        "-o /tmp/tmp1.xtc",
        "-pbc whole",
    ]
    if ignore_pbc == False:
        log.debug(f"{' '.join(cmd)}")
        try:
            subprocess.run(" ".join(cmd), shell=True, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            print(
                "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
            )
    # As step 1 is not run if ignore_pbc, set to original traj file in that case
    step2_input = trj_path if ignore_pbc else "/tmp/tmp1.xtc"
    # Step 2: -pbc cluster, produces tmp2.xtc
    cmd = [
        f"echo Protein {out_group} |",
        GMX,
        "trjconv",
        f"-f {step2_input}",
        f"-s {tpr}",
        f"-n {ndx}",
        "-o /tmp/tmp2.xtc",
        "-pbc cluster",
    ]
    log.debug(f"{' '.join(cmd)}")
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
        )
    # run trjconv to produce a readable output
    cmd = [
        f"echo Protein {out_group} |",
        GMX,
        "trjconv",
        "-f /tmp/tmp2.xtc",
        f"-s {tpr}",
        f"-n {ndx}",
        f"-o {out_path}",
        "-center",
    ]
    if ignore_pbc == False:
        cmd.append("-pbc mol -ur compact")
    else:
        cmd.append("-pbc cluster")
    log.debug(f"{' '.join(cmd)}")
    try:
        subprocess.run(" ".join(cmd), shell=True, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
        )
    # TODO -> REMOVE FILE
    # Remove temp xtc files if necessary
    subprocess.run("rm /tmp/*.xtc", shell=True)


def concat_traj(
    directory: str, stem: Optional[str] = None, out_path: str = "full_traj.xtc"
):
    # Assume input file extension based on output path
    ext = out_path.split(".")[-1]

    log.info(f"Concatenating Trajectories in {directory}/")

    # TODO: check that the files exist
    # TODO: check that all the names of the inputs are the same:
    #       i.e. there are not name.part000*.xtc AND name.xtc
    file_glob = f"{directory}/{stem}*.{ext}" if stem else f"{directory}/*.{ext}"
    cmd = [GMX, "trjcat", "-f", file_glob, "-o", f"{directory}/{out_path}"]
    log.debug(f"{' '.join(cmd)}")
    try:
        subprocess.run(" ".join(cmd), check=True, shell=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
        )


def snapshot_pdbs(
    trj_path: str,
    tpr: str,
    snapshots,
    ndx: str,
    out_group: str = "System",
    ns: bool = True,
    out_dir: Optional[str] = None,
    out_filename: Optional[str] = None,
) -> None:
    tpr = tpr if "/" in tpr else "/".join(trj_path.split("/")[:-1]) + "/" + tpr
    ndx = ndx if "/" in ndx else "/".join(trj_path.split("/")[:-1]) + "/" + ndx
    out_path = (
        "/".join(trj_path.split("/")[:-1]) + "/snapshots"
        if out_dir is None
        else out_dir
    )

    # TODO -> Make Dirs
    # Make the directory for the output
    try:
        subprocess.run(f"mkdir -p {out_path}", shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:", error.returncode, ". Output:", error.output.decode("utf-8")
        )

    # Define the output name
    stem = (
        trj_path.split("/")[-1].split(".")[0] if out_filename is None else out_filename
    )

    for ts in snapshots:
        out_ts = ts
        ts = ts * 1000 if ns else ts
        cmd = (
            f"echo {out_group} |gmx_mpi trjconv -f {trj_path} -s {tpr} -n {ndx} "
            f"-o {out_path}/{stem}_{out_ts}.pdb -dump {ts}"
        )
        try:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            print(
                "Error code:",
                error.returncode,
                ". Output:",
                error.output.decode("utf-8"),
            )


def run_driver(
    trj_path: str,
    top_path: str,
    driver_input: str,
    timestep: float = 0.002,
    stride: int = 1000,
    substitute: Optional[dict[str, str]] = None,
) -> None:
    top_path = (
        top_path
        if "/" in top_path
        else "/".join(trj_path.split("/")[:-1]) + "/" + top_path
    )
    assert trj_path[-4:] == ".xtc", "Trajectory must be of type XTC."
    assert top_path[-4:] == ".pdb", "Topology must be of type PDB."

    wd = "/".join(driver_input.split("/")[:-1])

    try:
        subprocess.run(
            (
                "plumed driver "
                f"--plumed {driver_input} "
                f"--pdb {top_path} "
                f"--mf_xtc {trj_path} "
                f"--trajectory-stride {stride} "
                f"--timestep {timestep}"
            ),
            cwd=wd,
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        print(
            "Error code:",
            error.returncode,
            ". Output:",
            error.output.decode("utf-8"),
        )

    """
    try:
        subprocess.run(f"mv loop_distance.dat {wd}/", shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print(
            "Error code:",
            error.returncode,
            ". Output:",
            error.output.decode("utf-8"),
        )

    """


def transfer_colvar_bias(
    old_colvar: str, new_colvar: str, out_path: Optional[str] = None
) -> None:
    clv1 = load.colvar(old_colvar)
    clv2 = load.colvar(new_colvar)

    out_path = new_colvar if out_path is None else out_path

    assert len(clv1.index) == len(
        clv2.index
    ), "Different length Colvars NOT SUPPORTED YET"

    clv2["meta.bias"] = clv1["meta.bias"]

    clv2 = clv2.astype(float)

    clv2 = clv2.drop("int_time", axis=1)

    output = [f"#! FIELDS {' '.join(list(clv2.columns))}\n"]
    for _, r in clv2.iterrows():
        output.append(
            f" {r.time:.6f} {' '.join([f'{r[field]:8.4f}' for field in list(clv2.columns)[1:]])}\n"
        )

    with open(out_path, "w") as f:
        f.writelines(output)
