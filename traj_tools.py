"""
===============================================================================
                                TRAJECTORY TOOLS
===============================================================================

    - PyTraj based analysis tools for Amber trajectories
"""

import numpy as np
import pytraj as pt
import subprocess
from glob import glob


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


def _run_cpptraj(directory, input_file):
    # starting message
    print(f"STARTING  | CPPTRAJ with input:  {input_file}")
    # run CPPTRAJ
    try:
        subprocess.run(f"cpptraj -i {directory}/{input_file}",
                       shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # message when finished successfully
    print(f"COMPLETED | CPPTRAJ with input:  {input_file}")


def make_fulltraj(directory, ref_str):
    # get file base name from dir. name
    stem = directory.split('/')[-1]
    # count the number of md steps run
    n_steps = len(glob(f"{directory}/{stem}.md_*.x"))
    # check stem is correct and files are found
    assert n_steps > 0, 'Not enough md files'
    # display info
    print(f" Making fulltraj for {stem} with {n_steps} steps.")
    file1 = []
    # load topology
    file1.append(f"parm {directory}/{stem}.top")
    # load each trajectory step
    for i in np.arange(n_steps)+1:
        file1.append(f"trajin {directory}/{stem}.md_{i}.x")
    # load reference structure (.top + .r)
    if isinstance(ref_str, list):
        file1.append(f"parm {ref_str[0]} [refparm]")
        file1.append(f"reference {ref_str[1]} parm [refparm]")
    # load reference structure (.pdb)
    else:
        file1.append(f"reference {ref_str}")
    # post-processing
    file1 += ['autoimage', 'rms reference @CA,C,N,O', 'strip :WAT,Na+,Cl-']
    # output
    file1.append(f"trajout {directory}/{stem}_{n_steps*5}ns_dry.nc netcdf")
    file1.append("go")
    # write all to cpptraj input file
    with open(f"{directory}/fulltraj.in", 'w+') as file:
        file.writelines('\n'.join(file1))
    # run cpptraj using that input file
    _run_cpptraj(directory, 'fulltraj.in')


def snapshot_pdbs(directory, trj_path, top_path, ref_str, snapshots):
    # make the out directory
    try:
        subprocess.run(f"mkdir -p {directory}/snapshots/",
                       shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    stem = trj_path.split('/')[-1].split('.')[0]
    if isinstance(snapshots[0], int):
        print('oops')
    elif isinstance(snapshots[0], list):
        for snl in snapshots:
            file1 = []
            file1.append(f"parm {top_path}")
            file1.append(f"trajin {trj_path} {' '.join([str(i) for i in snl])}")
            # load reference structure (.top + .r)
            if isinstance(ref_str, list):
                file1.append(f"parm {ref_str[0]} [refparm]")
                file1.append(f"reference {ref_str[1]} parm [refparm]")
            # load reference structure (.pdb)
            else:
                file1.append(f"reference {ref_str}")
            file1.append('rms reference @CA,C,N,O')
            file1.append(f"trajout {directory}/snapshots/{stem}.pdb multi chainid A")
            file1.append("go")
            # write all to cpptraj input file
            with open(f"{directory}/sn{snl[0]}.in", 'w') as file:
                file.writelines('\n'.join(file1))
            # run cpptraj using that input file
            _run_cpptraj(directory, f"sn{snl[0]}.in")
            snaps = np.arange(snl[0], snl[1], snl[2])
            for i in np.arange(len(snaps)):
                print(i, snaps[i])
                try:
                    subprocess.run(' '.join(['mv',
                                   f"{directory}/snapshots/{stem}.pdb.{i+1}",
                                   f"{directory}/snapshots/{stem}_{snaps[i]/200:.0f}ns.pdb"]),
                                   shell=True, check=True)
                except subprocess.CalledProcessError as error:
                    print('Error code:', error.returncode,
                          '. Output:', error.output.decode("utf-8"))


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
    # load the trajectory w. topology and run autoimage
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


def measure_rmsd(trj_path, top_path, ref_str, rmsd_mask,
                 aln_mask='@CA,C,N,O', nofit=True):
    # load the trajectory w. topology
    traj = pt.iterload(trj_path, top_path)
    # load ref. structure if path is given
    ref = _load_structure(ref_str)
    # run autoimage to cluster and centre traj.
    traj = traj.autoimage()
    # align the traj. using backbone atoms
    traj = pt.align(traj, mask=aln_mask, ref=ref)
    # calculate rmsd
    data = pt.rmsd(traj, mask=rmsd_mask, ref=ref, nofit=nofit)
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
    # load the trajectory w. topology and run autoimage
    full_trj = pt.iterload(trj_path, top)
    full_trj = full_trj.autoimage()
    print(f'Loaded trajectory: {trj_path}')
    # calculate frame to extract
    N = int(full_trj.n_frames)-1 if frame == 'final' else int(frame)-1
    print(f'Writing {N+1}th frame as pbd')
    # save the new trajectory file
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
    # check that correct atom/res mask is defined
    n_atoms = len(angle_atoms.split())
    assert n_atoms in [3, 4], "INPUT ERROR: Must have 3 or 4 atom groups."
    # load the trajectory w. topology
    traj = pt.iterload(trj_path, top_path)
    # load ref. structure if path is given
    ref = _load_structure(ref_str)
    # run autoimage to cluster and centre traj.
    traj = traj.autoimage()
    # align the traj. using backbone atoms
    traj = pt.align(traj, mask=aln_mask, ref=ref)
    # calculate angle between set of atoms
    if n_atoms == 3:
        print("Calculating angle between 3 atom groups.")
        data = pt.angle(traj=traj, mask=angle_atoms)
    else:
        print("Calculating dihedral angle between 4 atom groups.")
        data = pt.dihedral(traj=traj, mask=angle_atoms)
    return data
