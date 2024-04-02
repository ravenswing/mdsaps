"""
===============================================================================
                             TRAJECTORY TOOLS
===============================================================================

    - MDAnalysis based tools for analysing trajectories
    - File conversion (amber <-> gromacs <-> pdb)

    N.B. Requires ambertools (PyTraj & Parmed) for file conv.
"""

import logging
import MDAnalysis as mda
import numpy as np
import pandas as pd
import pickle
import pytraj as pt
import subprocess
from multiprocessing import Pool
from functools import partial
from numbers import Number
from MDAnalysis.analysis import rms
from parmed import gromacs, amber, load_file
from os.path import exists

log = logging.getLogger(__name__)
log.info("Traj Tools Loaded")


def _process_atm_nm(name: str) -> str:
    log.debug(f'Atom Name: |{name}|')
    # Full length names are left unchanged
    if len(name) == 4:
        return name
    # Otherwise align to second character
    else:
        return f" {name:<3}"


def format_pdb(info: list, chain: bool = False) -> str:
    # Process atom name seperately
    atm_nm = _process_atm_nm(info[2])

    # Add chain
    c = 1 if chain else 0

    # Assign values for each column
    record = info[0]  # Record name
    atm_id = info[1]  # Atom serial number
    alt_li = ' '  # Alternate location indicator
    res_nm = info[3]  # Residue name.
    chn_id = ' ' if not chain else info[4]  # Chain ID
    res_id = info[4 + c]  # Residue sequence number
    i_code = ' '   # Code for insertion
    coords = [float(info[5 + c]),  # x (A)
              float(info[6 + c]),  # y (A)
              float(info[7 + c])]  # z (A)
    occupn = float(info[8 + c])  # Occupancy
    temprt = float(info[9 + c])  # Temperature
    elemnt = info[10 + c]  # Element
    charge = '  '  # Charge

    # Format the new line using all the values
    new_line = (f"{record:6}"
                f"{atm_id:>5} "
                f"{atm_nm:4}"
                f"{alt_li}"
                f"{res_nm:3} "
                f"{chn_id}"
                f"{res_id:>4}"
                f"{i_code}   "
                f"{coords[0]:>8.3f}{coords[1]:>8.3f}{coords[2]:>8.3f}"
                f"{occupn:>6.2f}"
                f"{temprt:>6.2f}          "
                f"{elemnt:>2}"
                f"{charge}\n")
    log.debug(f'Format PBD: |{new_line}')

    # Return the new line
    return new_line


def amber_to_gromacs(top_file: str, crd_file: str) -> None:
    ''' Convert a system from Amber --> Gromacs using ParmEd '''
    # Check that the topology has a readable extension
    assert top_file.split('.')[-1] in ['parm7', 'prmtop'], "ERROR"
    # Check that the coordinate file has a readable extension
    assert crd_file.split('.')[-1] in ['rst7', 'ncrst', 'restrt'], "ERROR"

    # Load the system (from ParmEd)
    amber = load_file(top_file, crd_file)
    # Write the new Gromacs topology file (.top)
    amber.save(f"{top_file.split('.')[0]}_a2g.top", overwrite=True)
    # Write the new Gromacs coordinate file (.gro)
    amber.save(f"{crd_file.split('.')[0]}_a2g.gro", overwrite=True)


def gromacs_to_amber(top_file: str, crd_file:str) -> None:
    ''' Convert a system from Gromacs --> Amber using ParmEd '''
    # Check that the topology has a readable extension
    assert top_file.split('.')[-1] == 'top', "ERROR"
    # Check that the coordinate file has a readable extension
    assert crd_file.split('.')[-1] == 'gro', "ERROR"

    # Import the Gromacs topology (gromacs from ParmEd)
    gmx_top = gromacs.GromacsTopologyFile(top_file)
    # Import the Gromacs coordinates
    gmx_gro = gromacs.GromacsGroFile.parse(crd_file)
    # Exchange some of the information as Amber/Gromacs files store dif. info
    gmx_top.box = gmx_gro.box  # (Needed because .prmtop contains box info)
    gmx_top.positions = gmx_gro.positions
    # Create Amber parm object (amber from ParmEd)
    amb_prm = amber.AmberParm.from_structure(gmx_top)
    # Write the new Amber topology file (.prmtop)
    amb_prm.write_parm(f"{top_file.split('.')[0]}_g2a.prmtop")
    # Write the new Amber coordinate file (.rst7)
    amb_crd = amber.AmberAsciiRestart(f"{crd_file.split('.')[0]}_g2a.rst7",
                                      mode="w")
    amb_crd.coordinates = gmx_top.coordinates
    amb_crd.box = gmx_top.box
    amb_crd.close()


def amber_to_pdb(top_file: str, crd_file: str, autoimage: bool = False) -> None:
    ''' Convert a system from Amber --> PDB using PyTraj '''
    # Check that the topology has a readable extension
    assert top_file.split('.')[-1] in ['parm7', 'prmtop'], "ERROR"
    # Check that the coordinate file has a readable extension
    assert crd_file.split('.')[-1] in ['rst7', 'ncrst', 'restrt'], "ERROR"

    # Load the amber structure into PyTraj
    to_convert = pt.load(crd_file, top_file)
    if autoimage:
        to_convert = to_convert.autoimage()
    # Write the new .pdb file
    pt.write_traj(f"{crd_file.split('.')[0]}_a2p.pdb",
                  to_convert,
                  overwrite=True)


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


def multiindex_hdf(new_data, ids, hdf_path, data_col, index_col):

    # Create dataframe from new data with desired index and 1 column
    inp = new_data.rename(columns={data_col: ids[-1]}).set_index(index_col)
    # Create multi-indexed dataframe based on provided ids (as levels)
    new_dfs = []
    for i, level in enumerate([ids[i] for i in range(len(ids) - 2, -1, -1)]):
        # First instance takes the input dataframe made above
        if i == 0:
            new_dfs.append(pd.concat({level: inp}, axis=1))
        # Others take the ids in reverse order, and the prev. df
        else:
            new_dfs.append(pd.concat({level: new_dfs[i-1]}, axis=1))
    # Multi-Indexed df is the last df made
    df = new_dfs[-1]

    # If there is already an hdf file...
    if exists(hdf_path):
        new = pd.read_hdf(hdf_path, key='df')
        # ...and the column exists, update the values.
        if any([(mi == df.columns)[0] for mi in new.columns]):
            log.info('HDF Exists -> Updating values in DataFrame.')
            new.update(df)
        # ...and the data is new, add the new data.
        else:
            log.info('HDF Exists -> Adding new values to DataFrame.')
            new = new.join(df)
        # Reorder the columns before saving the data.
        new = new.iloc[:, new.columns.sortlevel(0, sort_remaining=True)[1]]
        # Write the new data to the existing file.
        new.to_hdf(hdf_path, key='df')
    # But if there is not a file already...
    else:
        # ... make a new hdf file and save the first column of data.
        log.info('No HDF Found -> Creating File')
        df.to_hdf(hdf_path, key='df')


def measure_rmsd(top_path, trj_path, ref_str, rmsd_groups,
                 aln_group='backbone'):
    # Load the topology and trajectory
    U = _init_universe([top_path, trj_path])
    if ref_str:
        # Load ref. structure if path is given
        ref = _init_universe(ref_str)
    else:
        # If ref_str = 0 i.e. use starting frame, assign ref as input traj.
        ref = U
    R = rms.RMSD(U,  # universe to align
                 ref,  # reference universe or atomgroup
                 select=aln_group,  # group to superimpose and calculate RMSD
                 groupselections=rmsd_groups,  # groups for RMSD
                 ref_frame=0).run()  # frame index of the reference
    return pd.DataFrame(columns=['t', 'rmsd'], data=R.results.rmsd[:, [1, -1]])


def save_rmsd(ids, top_path, trj_path, hdf_path, measure,
              ref_path=None, align='backbone'):
    log.info(f"Running RMSD Calc. for {' '.join(ids)}")
    rmsd = measure_rmsd(top_path, trj_path, ref_path,
                        [measure], aln_group=align)
    multiindex_hdf(rmsd, ids, hdf_path, 'rmsd', 't')


def dump_rmsd(top_path, trj_path, ref_str, out_path=None,
              align='backbone', measure=['backbone']):

    R = measure_rmsd(top_path, trj_path, ref_str, measure, aln_group=align)

    if out_path:
        print(f"Writing RMSD file to: {out_path}")
        rmsd = R.results.rmsd[:, -1]
        with open(out_path, 'wb') as f:
            pickle.dump(rmsd, f)
    else:
        for i, name in enumerate(measure):
            rmsd = R.results.rmsd[:, 3+i]
            outname = f"{'/'.join(trj_path.split('/')[:-1])}/{name}_rmsd.p"
            print(f"Writing RMSD file to: {outname}")
        with open(outname, 'wb') as f:
            pickle.dump(rmsd, f)


def measure_rmsf(top_path, trj_path, measure='backbone',
                 select='protein', per_res=True):
    res = '-res' if per_res else ''
    # Get the directory that this file is in.
    s_path = '/'.join(__file__.split('/')[:-1])
    cmd = ["python", f"{s_path}/measure_rmsf.py",
           top_path, trj_path,
           '/tmp/rmsf.h5',
           f"\"{measure}\"", f"\"{select}\"", res]
    # Measure the RMSF
    try:
        subprocess.run(' '.join(cmd), shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

    df = pd.read_hdf('/tmp/rmsf.h5', key='df', mode='r').reset_index()
    return df


def save_rmsf(ids, top_path, trj_path, hdf_path, measure='backbone',
              select='protein', per_res=True):
    log.info(f"Running RMSF Calc. for {' '.join(ids)}")
    rmsf = measure_rmsf(top_path, trj_path, measure, select, per_res)
    index = 'res' if per_res else 'atom'
    multiindex_hdf(rmsf, ids, hdf_path, 'rmsf', index)


def simple_rmsf(top_path, trj_path, ref_str, rmsd_groups,
                aln_group='backbone'):
    # Load the topology and trajectory
    U = _init_universe([top_path, trj_path])
    if ref_str:
        # Load ref. structure if path is given
        ref = _init_universe(ref_str)
    else:
        # If ref_str = 0 i.e. use starting frame, assign ref as input traj.
        ref = U
    R = rms.RMSF(U,  # universe to align
                 ref,  # reference universe or atomgroup
                 select=aln_group,  # group to superimpose and calculate RMSD
                 groupselections=rmsd_groups,  # groups for RMSD
                 ref_frame=0).run()  # frame index of the reference
    return R


def measure_rgyr(top_path, trj_path, selection):
    # Load the topology and trajectory
    U = _init_universe([top_path, trj_path])
    Rgyr = []
    protein = U.select_atoms(selection)
    for ts in U.trajectory:
        Rgyr.append((U.trajectory.time, protein.radius_of_gyration()))
    Rgyr = np.array(Rgyr)
    return pd.DataFrame(columns=['t', 'rgyr'], data=Rgyr)


def save_rgyr(ids, top_path, trj_path, hdf_path, measure, align='backbone'):
    log.info(f"Running Rad. Gyr. Calc. for {' '.join(ids)}")
    rgyr = measure_rgyr(top_path, trj_path, measure)
    multiindex_hdf(rgyr, ids, hdf_path, 'rgyr', 't')


def calc_3D_dist(x1: Number, y1: Number, z1: Number,
                 x2: Number, y2: Number, z2: Number):
    return np.sqrt(np.square(x2-x1) + np.square(y2-y1) + np.square(z2-z1))


def com_dist(idx, u, atom_groups):
    u.trajectory[idx]
    # Split the trajectores to just portal & ligand atoms
    comA = u.select_atoms(atom_groups[0]).center_of_mass()
    comB = u.select_atoms(atom_groups[1]).center_of_mass()
    dist = calc_3D_dist(comA[0], comA[1], comA[2],
                        comB[0], comB[1], comB[2],)
    return dist


def measure_com_dist(top_path: str, trj_path: str, selectA: str, selectB: str,
                     indices=None):
    u = _init_universe([top_path, trj_path])
    indices = np.arange(u.trajectory.n_frames) if indices is None else indices
    run_per_frame = partial(com_dist,
                            u=u,
                            atom_groups=[selectA, selectB])
    with Pool(4) as worker_pool:
        result = worker_pool.map(run_per_frame, indices)
    result = np.asarray(result).T
    return pd.DataFrame.from_dict({'t': indices, 'com': result})


def save_com_dist(ids: list, top_path: str, trj_path: str, hdf_path: str,
                  selectA: str, selectB: str, indices=None):
    log.info(f"Running COM Dist. Calc. for {' '.join(ids)}")
    dist = measure_com_dist(top_path, selectA, selectB, indices)
    multiindex_hdf(dist, ids, hdf_path, 'com', 't')


def atom_numbers(pdb, select, names=None):
    """
        Extract the atom numbers from a topology, based on selection.
        If names are given, are chained with 'or' to the selection.
    """
    # Load the pdb into MDAnalysis
    u = _init_universe(pdb)
    # Create the string to pass to select (adding names as a list of 'or's).
    sel_str = f"{select} and (name {' '.join(names)})"
    log.debug(f"Select str used: {sel_str}")
    # Select the AtomGroup with just the wanted atoms.
    u = u.select_atoms(sel_str)
    # Pass the IDs (from pdb input) as a list.
    return list(u.atoms.ids)


def simple_avg_table(hdf_path, csv=None):
    df = pd.read_hdf(hdf_path, key='df')
    df.columns = df.columns.map(','.join)
    out_list = []
    for col, data in df.items():
        line = [col,
                f"{data.mean():.4f}",
                f"{data.std():.4f}"]
        line = ','.join(line)
        out_list.append(line)
    if csv:
        print(f"Saved to file {csv}")
        with open(csv, 'w') as f:
            f.writelines([ln + '\n' for ln in out_list])
    else:
        for ln in out_list:
            print(ln)


def comb_mean_std(N, X, S):
    # the mean of total group is : (n1*X1+n2*X2)/(n1+n2)
    # the variance of total group is : n1*(S1^2+d1^2)+n2*(S22+d22)/(n1+n2)
    # where  d1 = X1-mean of total group

    comb_mean = np.multiply(N, X).sum()/N.sum()
    var = np.square(S)
    div2 = np.square(np.subtract(X, comb_mean))
    temp = np.add(var, div2)
    temp = np.multiply(temp, N)
    comb_std = np.sqrt(temp.sum()/N.sum())

    return comb_mean, comb_std


def usym(string: str) -> str:
    """
        Encode unicode symbols based on standard naming...
    """
    string = string.casefold()
    # Angstrom 
    if "ang" in string or string == "a":
        u_char = "\u212B"
    # Plus Minus
    elif string == "pm" or all(x in string for x in ["pl", "mi"]):
        u_char = "\u00B1"
    # Degress
    elif "deg" in string:
        u_char = "\u00B0"
    # Greek letter alpha (lower case)
    elif any(x in string for x in ["alpha", "al"]):
        u_char = "\u03B1"
    # Greek letter beta (lower case)
    elif any(x in string for x in ["beta", "be"]):
        u_char = "\u03B2"
    # Greek letter gamma (lower case)
    elif any(x in string for x in ["gamma", "ga", "y"]):
        u_char = "\u03B3"
    else:
        raise ValueError('Please input a valid symbol reference.')
    return u_char
