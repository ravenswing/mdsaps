"""
===============================================================================
                         TRAJECTORY ANALYSIS TOOLS
===============================================================================

    - MDAnalysis based tools for analysing trajectories
"""

import pytraj as pt
import numpy as np
from parmed import gromacs, amber, load_file
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import pickle
from itertools import product
import pandas as pd

ANG = "\u212B"


def read_pdb(path):
    # Load complex pdb to edit
    with open(path, 'r') as f:
        lines = f.readlines()
    print(f'Loaded {path}')
    lines = [line.split() for line in lines]
    return lines


def _process_atm_nm(name):
    # Full length names are left unchanged
    if len(name) == 4:
        return name
    # Otherwise align to second character
    else:
        return f" {name:<3}"


def format_pdb(info, chain=False):
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

    # Return the new line
    return new_line


def amber_to_gromacs(top_file, crd_file):
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


def gromacs_to_amber(top_file, crd_file):
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


def amber_to_pdb(top_file, crd_file, autoimage=False):
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
    return R


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


def calculate_rmsd(DIVS, top_frmt, trj_frmt, hdf_path, measure,
                   ref_frmt=None, align='backbone', unique_ligs=False):
    all_sys = DIVS if unique_ligs else product(*DIVS)
    for i, p in enumerate(all_sys):
        p = list(p)

        print(p)
        print(top_frmt.format(p=p))
        print(trj_frmt.format(p=p))

        ref = ref_frmt if ref_frmt else top_frmt

        print(ref)

        # LIGAND & BACKBONE RMSD
        new_data = measure_rmsd(top_frmt.format(p=p),
                                trj_frmt.format(p=p),
                                ref.format(p=p),
                                [measure],
                                aln_group=align).run()
        if len(all_sys[0]) == 3:

            inp = pd.DataFrame(columns=['t', p[2]],
                    data=new_data.results.rmsd[:, [1, -1]]).set_index('t')
            inp_l = pd.concat({p[1]: inp}, axis=1)
            inp_s = pd.concat({p[0]: inp_l}, axis=1)

        elif len(all_sys[0]) == 2:
            inp = pd.DataFrame(columns=['t', p[1]],
                    data=new_data.results.rmsd[:, [1, -1]]).set_index('t')
            inp_s = pd.concat({p[0]: inp}, axis=1)

        if i == 0:
            print('First time --> Creating Files')
            inp_s.to_hdf(hdf_path, key='df')

            i += 1
            continue
        print('Further time --> Reading Files & Adding Data')
        new = pd.read_hdf(hdf_path, key='df')

        if any([(mi == inp_s.columns)[0] for mi in new.columns]):
            print("Updating values in DataFrame.")
            new.update(inp_s)
        else:
            print("Adding new values to DataFrame.")
            new = new.join(inp_s)
        # reorder columns
        new = new.iloc[:, new.columns.sortlevel(0, sort_remaining=True)[1]]
        new.to_hdf(hdf_path, key='df')


def measure_rgyr(top_path, trj_path, selection):
    # Load the topology and trajectory
    U = _init_universe([top_path, trj_path])
    Rgyr = []
    protein = U.select_atoms(selection)
    for ts in U.trajectory:
        Rgyr.append((U.trajectory.time, protein.radius_of_gyration()))
    Rgyr = np.array(Rgyr)
    return Rgyr


def calculate_rgyr(DIVS, top_frmt, trj_frmt, hdf_path, measure='protein'):

    for i, p in enumerate(product(*DIVS)):
        p = list(p)

        # LIGAND & BACKBONE RMSD
        new_data = measure_rgyr(top_frmt.format(p=p),
                                trj_frmt.format(p=p),
                                measure)
        if len(DIVS) == 3:

            inp = pd.DataFrame(columns=['t', p[2]],
                    data=new_data).set_index('t')
            inp_l = pd.concat({p[1]: inp}, axis=1)
            inp_s = pd.concat({p[0]: inp_l}, axis=1)

        elif len(DIVS) == 2:
            inp = pd.DataFrame(columns=['t', p[1]],
                    data=new_data.results.rmsd[:, [1, -1]]).set_index('t')
            inp_s = pd.concat({p[0]: inp}, axis=1)

        if i == 0:
            print('First time --> Creating Files')
            inp_s.to_hdf(hdf_path, key='df')

            i += 1
            continue
        print('Further time --> Reading Files & Adding Data')
        new = pd.read_hdf(hdf_path, key='df')

        if any([(mi == inp_s.columns)[0] for mi in new.columns]):
            print("Updating values in DataFrame.")
            new.update(inp_s)
        else:
            print("Adding new values to DataFrame.")
            new = new.join(inp_s)
        # reorder columns
        new = new.iloc[:, new.columns.sortlevel(0, sort_remaining=True)[1]]
        new.to_hdf(hdf_path, key='df')
