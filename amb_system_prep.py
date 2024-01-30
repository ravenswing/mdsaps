"""
===============================================================================
                              AMBER SYSTEM TOOLS
===============================================================================

        Required inputs:
            - PDB of apo protein
            - PDB of ligand
            - Ligand Parameters: .frcmod & .prep
"""


import numpy as np
import pandas as pd
import pytraj as pt
import subprocess
from glob import glob

SCRIPT_DIR = ("/home/rhys/phd_tools/simulation_files/"
              "submission_scripts/Amber/md")


def _run_tleap(wd, input_file):
    # Print a starting message
    print(f"STARTING  | TLEAP with input:  {input_file}")
    # Run TLEAP
    try:
        o = subprocess.run(f"tleap -f {wd}/{input_file}",
                           shell=True, check=True,
                           capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    with open(f"{wd}/leap.log", 'w') as f:
        f.write(o.stdout)
    # Print another message when finished successfully
    print(f"COMPLETED | TLEAP with input:  {input_file}")


def build_system(wd, lig_param_path, complex_path, out_name):
    # Add final protein res. no. to min0 input (N.B. inplace!)
    with open(f"{SCRIPT_DIR}/template_tleap.in", 'r') as file:
        lines = file.read()
        lines = lines.replace('LIG_PREP', lig_param_path+'.prep')
        lines = lines.replace('LIG_FRCMOD', lig_param_path+'.frcmod')
        lines = lines.replace('COMPLEX_PDB', complex_path)
        lines = lines.replace('OUTNAME', out_name)
    with open(f"{wd}/build.tleap", 'w') as file:
        file.write(lines)

    _run_tleap(wd, 'build.tleap')


def setup_minimisation(wd, pdb_filename, nm_pos_in_path, restraints=False):
    '''
        1. Creates the directory for minimisation
        2. Copies in the template input scripts
        3. Edits the Protein residue no. in min0
        4. Edits the Water residue no.s in min1
        5. Adds the filename to minim.sh
    '''
    SCRIPT_DIR = ("/home/rhys/phd_tools/simulation_files/"
                  "submission_scripts/Amber/md")
    # Make the directory for the minimisation.
    try:
        subprocess.run(f"mkdir -p {wd}", shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Locate the template scripts
    if restraints:
        SCRIPT_DIR += "_restraints"
    # Copy the templates to the working directory
    try:
        subprocess.run(' '.join(['cp', f"{SCRIPT_DIR}/min*", wd+'/']),
                       shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Read in pdb to extract residue numbers from
    with open(f"{wd}/{pdb_filename}", 'r') as f:
        pdb = f.readlines()
    # Identify lines within pdb that contain TER
    i_ter = []
    for i in np.arange(len(pdb)):
        if 'TER' in pdb[i]:
            i_ter.append(i)
    # Final protein res. no. (line before first line with TER)
    prt_end = pdb[i_ter[0]-1].split()[4]
    # First water res. no. (first line with WAT)
    wat_str = [line.split()[4] for line in pdb if 'WAT' in line][0]
    # Last water res. no. (line before last line with TER)
    wat_end = pdb[i_ter[-1]-1].split()[4]
    # Add final protein res. no. to min0 input (N.B. inplace!)
    with open(f"{wd}/min0.in", 'r') as file:
        lines = file.read()
        lines = lines.replace("PROT_END", prt_end)
    with open(f"{wd}/min0.in", 'w') as file:
        file.write(lines)
    # Add water res. no. to min1 input (N.B. inplace!)
    with open(f"{wd}/min1.in", 'r') as file:
        lines = file.read()
        lines = lines.replace("WATER_RES", f"{wat_str} {wat_end}")
    with open(f"{wd}/min1.in", 'w') as file:
        file.write(lines)
    file_stem = wd.split('/')[nm_pos_in_path]
    # Change the name in the running bash script (N.B. inplace!)
    with open(f"{wd}/minim.sh", 'r') as file:
        lines = file.read()
        lines = lines.replace("FILE_STEM_HERE", f"{file_stem}")
    with open(f"{wd}/minim.sh", 'w') as file:
        file.write(lines)


def make_restraints_list(wd, txt_in):
    # Create new list (IS THIS NEEDED???)
    new_file = ['C=O']
    # Read in restraints .csv and select the correct set
    df = pd.read_table(txt_in, sep=",", header=0)
    df = df.loc[df.complex == f"{FN}_dry"]
    # Create a block for each restraint being applied from the .csv values
    for z, restr in df.iterrows():
        # name of restraint
        nm_line = f"# Interaction {z}"
        # variable parameters
        ir_line = (f"  ixpk= 0, nxpk= 0, iat={restr.i1},{restr.i2}, "
                   f"r1= {restr.cp - 0.6:.1f}, "
                   f"r2= {restr.cp - 0.2:.1f}, "
                   f"r3= {restr.cp + 0.2:.1f}, "
                   f"r4= {restr.cp + 0.6:.1f},")
        # put all in block with invariant lines
        restr_block = ['#', nm_line, ' &rst', ir_line,
                       '      rk2=10.0, rk3=10.0, ir6=1, ialtd=0,', ' &end']
        new_file += restr_block
    # Write the new list to file
    with open(f"{wd}/restraints.list", 'w+') as f:
        f.write('\n'.join(new_file))


def run_minimisation(wd):
    # Exectute the bash script for minimisation
    try:
        subprocess.run(' '.join(['bash', f"{wd}/minim.sh"]), check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def make_reducing_restraints(wd, txt_in):
    for k in [10, 5, 2.5]:
        # Create new list (IS THIS NEEDED???)
        new_file = ['C=O']
        # Read in the restraints .csv and select the correct set
        df = pd.read_table(txt_in, sep=",", header=0)
        df = df.loc[df.complex == f"{FN}_dry"]
        # Create block for each restraint being applied using numbers from .csv
        for z, restr in df.iterrows():
            # name of restraint
            nm_line = f"# Interaction {z}"
            # variable parameters
            ir_line = (f"  ixpk= 0, nxpk= 0, iat={restr.i1},{restr.i2}, "
                       f"r1= {restr.cp - 0.6:.1f}, "
                       f"r2= {restr.cp - 0.2:.1f}, "
                       f"r3= {restr.cp + 0.2:.1f}, "
                       f"r4= {restr.cp + 0.6:.1f},")
            # put all in block with invariant lines
            restr_block = ['#', nm_line,
                           ' &rst', ir_line,
                           f"      rk2={k:.1f}, rk3={k:.1f}, ir6=1, ialtd=0,",
                           ' &end']
            new_file += restr_block
        # Write the new list to file
        with open(f"{wd}/restraints_k{k:.0f}.list", 'w+') as f:
            f.write('\n'.join(new_file))


def transfer(wd):
    path = '/'.join(wd.split('/')[-2:])
    print(path)
    try:
        subprocess.run(' '.join(['rsync -avzhPu',
                                 f"{wd}/restraints_k*",
                                 f"{SVR_DIR}/{path}/"]),
                       shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def autoimage_file(top_file, crd_file):
    ''' Convert a system from Amber --> PDB using PyTraj '''
    # Check that the topology has a readable extension
    assert top_file.split('.')[-1] in ['parm7', 'prmtop'], "ERROR"
    # Check that the coordinate file has a readable extension
    assert crd_file.split('.')[-1] in ['rst7', 'ncrst', 'restrt'], "ERROR"

    # Load the amber structure into PyTraj
    to_convert = pt.load(crd_file, top_file)
    to_convert = to_convert.autoimage()
    # Write the new .pdb file
    out_name = f"{crd_file.split('.')[0]}_ai.rst7"
    pt.write_traj(out_name,
                  to_convert,
                  options="keepext",
                  overwrite=True)


if __name__ == "main":

    # lig21
    POCKETS = ['Tunnel-Front', 'Tunnel-Back', 'Active-Site']
    DATA_DIR = ('/home/rhys/Dropbox/RESEARCH/AA_RHYS/BB_BECK/'
                'SHIP_uMD_Prep/Lig21_uMD')

    # MINIMISATION
    for pocket in POCKETS:
        wd = f"{DATA_DIR}/{pocket.lower()}"
        make_restraints_list(wd,
                             f"complex_ship2_{pocket}+Lig21",
                             f"{DATA_DIR}/SHIP Project - Lig21 Restraints.csv")
        prepare_min_inputs(wd, restraints=True)

    # PRODUCTION
    POCKETS = ['Experimental', 'Tunnel-Front', 'Tunnel-Back']
    DATA_DIR = '/media/rhys/data1/ship_rhys_2022/holo_uMD/minimisation'
    SVR_DIR = 'iqtc:/home/g19torces/rhys_running/ship_holo_uMD'

    for pocket in POCKETS:
        for path in glob(f"{DATA_DIR}/{pocket.lower()}/*"):
            FN = f"complex_{path.split('/')[-1]}"
            make_reducing_restraints(path, '/home/rhys/SHIP/Data/SHIP Project - Restraints.csv')
            transfer(path)
    #         prepare_min_inputs(path, restraints=True)
