"""
===============================================================================
                            GROMACS SYSTEM PREPARATION
===============================================================================

        Required inputs:
            - PDB of apo protein
            - PDB of ligand
            - Ligand Parameters: .frcmod & .prep
"""


import subprocess
from itertools import chain
import numpy as np
import parmed as pmd
import pytraj as pt

PREP_INPUTS = '/home/rhys/phd_tools/simulation_files/submission_scripts/Local_Dirs/00-Prep'
OUT_DIR = '/home/rhys/Storage/ampk_metad_all_data'
PARM_DIR = '/home/rhys/AMPK/Metad_Simulations/System_Setup/ligand_parms'
SCRIPT_DIR = '/home/rhys/phd_tools/simulation_files/submission_scripts/MareNostrum/class_a'
REMOTE = 'mn:/home/ub183/ub183944/scratch/ampk_replicas'

SYSTS = ['a2b1', 'a2b2']
#SYSTS = ['a2b2']
#LIGS = [ 'A769']
#LIGS = ['SC4', 'PF739', 'MT47', 'MK87']
LIGS = ['MK87', 'MT47']


def make_dirs(name, sys, lig):
    # Make the working directory
    try:
        subprocess.call(['mkdir', "-p", name])
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Copy in the apo pdb
    try:
        subprocess.call(['cp', f"/home/rhys/AMPK/Metad_Simulations/System_Setup/apo/{sys}_apo.pdb", name])
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Copy in the ligand pdb
    try:
        subprocess.call(['cp', f"/home/rhys/AMPK/Metad_Simulations/System_Setup/ligand_pdbs_4_gmx/{lig}_4_{sys}_4_gmx.pdb", name])
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def check_atom_order(pdb_file, prep_file):
    # Open both the .pdb and .prep files
    with open(pdb_file, 'r') as f:
        pdb = f.readlines()
    with open(prep_file, 'r') as f:
        prep = f.readlines()
    # Clean files and isolate atom names
    atm_ord1 = [line.split()[2] for line in pdb
                if not any(x in line for x in ['TER', 'END'])]
    atm_ord2 = [line.split()[1] for line in prep
                if len(line.split()) >= 7 and 'DUMM' not in line]
    # Atom names must be idential 
    return atm_ord1 == atm_ord2


def run_tleap(lig, pdb_path):

    lig_params = f"{PARM_DIR}/{lig}"

    ffi_lines = 'source leaprc.gaff'
    lig_lines = f"loadamberparams {lig_params}.frcmod\nloadamberprep {lig_params}.prep"
    pdb_lines = f"struc = loadpdb {pdb_path}"
    out_lines = f"saveamberparm struc {wd}/{lig}_amb.top {wd}/{lig}_amb.crd\nsavepdb struc {wd}/{lig}_leap.pdb"

    with open('./temp.tleap', 'w+') as f:
        f.write('\n'.join([ffi_lines, lig_lines,
                           pdb_lines, out_lines, 'quit']))

    try:
        subprocess.call(['rm', './leap.log'])
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

    try:
        subprocess.call(['tleap', '-f ./temp.tleap'])
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

#     try:
#         sub = subprocess.Popen('tleap -f ./temp.tleap',
#                                stdout=subprocess.PIPE,
#                                stderr=subprocess.STDOUT,
#                                shell=True
#                               )
#         out,errors =  sub.communicate()
#     except:
#         print("ERROR: TLeap unable to complete")


def run_parmed(prmtop, crd, out_path):
    amber = pmd.load_file(prmtop, crd)
    amber.save(f"{out_path}.top")
    amber.save(f"{out_path}.gro")


def run_pdb2gmx(pdb_file, out_name):
    try:
        sub = subprocess.Popen(f"gmx_mpi pdb2gmx -f {pdb_file} -o {out_name}.gro -p {out_name}.top -i {out_name}_posre.itp -ff amber14sb_gmx_s2p -water tip3p -ignh",
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True
                               )
        out, errors = sub.communicate()
    except:
        print("ERROR: TLeap unable to complete")


def combine_gro(prot_path, lig_path, out_path=None):
    # Load the protein gro file
    with open(prot_path, 'r') as f:
        protein_gro = f.readlines()
    # Load the ligand gro file
    with open(lig_path, 'r') as f:
        ligand_gro = f.readlines()
    # Find the total number of atoms that the new gro will have
    total = (int(ligand_gro[1].split('\n')[0]) + int(protein_gro[1].split('\n')[0]))
    print(total)
    # Save that total as the start of the document
    line2 = ' '+str(total)+'\n'
    # Combine the the new total, the protein and ligand gro files 
    both_gro = chain(protein_gro[0], line2, protein_gro[2:-1], ligand_gro[2:-1], protein_gro[-1])
    # Write the combined file to a new or custom path
    out_name = out_path if out_path else f"{'/'.join(lig_path.split('/')[:-1])}/{(prot_path.split('/')[-1][:-4])}+{(lig_path.split('/')[-1][:-4])}_built.gro"
    with open(out_name, 'w+') as f:
        f.writelines(str(line) for line in both_gro)


def combine_top(prot_top, lig_top, lig_name=None, directory=None):
    # Set the output path if not pre-defined
    out_path = directory if directory else f"{'/'.join(lig_top.split('/')[:-1])}"
    # Load the protein .top file
    with open(prot_top) as f:
        pro = f.readlines()
    # Load the ligand .top file
    with open(lig_top) as f:
        d = '['
        lig = [d+e for e in f.read().split(d) if e]
    # Put ligand atomtypes in a separate .itp file
    with open(f"{out_path}/ligand_atomtypes.itp", 'w') as f:
        f.write([st for st in lig if 'atomtypes' in st][0])
    # Remove all necessary sections from ligand topology
    remove = ['defaults', 'atomtypes', 'molecules', 'system']
    include = '\n'.join([st for st in lig[1:]
                         if not any(x in st for x in remove)])
    # Write the remainder to the ligand itp file
    with open(f"{out_path}/ligand.itp", 'w+') as f:
        f.write(include)
    # Extract the ligand residue name from the ligand topology
    lig_res_name = lig_name if lig_name else [
        st for st in lig if 'moleculetype' in st][0].split('\n')[2].split()[0]
    print(f"Using Ligand: {lig_res_name}")
    # Define new lines to add to prot topology, with include lines and comments
    include1 = '; Include ligand atom types \n#include \"./ligand_atomtypes.itp\" \n\n'
    include2 = '; Include ligand topology   \n#include \"./ligand.itp\" \n\n'
    # Also a line for the very bottom of the file
    # to add to the [molecules] entry
    include3 = f'{lig_res_name}                 1\n'
    # Atomtypes must appear before any [moleculetype] entry
    #    in this case in the chain topologies
    n1 = [i for i, s in enumerate(pro) if 'chain' in s][0]
    # Rest of lig. topology then goes before the water topology is loaded
    n2 = [i for i, s in enumerate(pro) if 'water topology' in s][0]
    # Put the new lines in the correct place in the file
    new_top = chain(pro[:n1], include1,
                    pro[n1:n2], include2,
                    pro[n2:], include3)
    # Write the combined file to a new or custom path
    out_name = f"{out_path}/{(prot_top.split('/')[-1][:-4])}+{(lig_top.split('/')[-1][:-4])}.top"
    with open(out_name, 'w+') as f:
        f.writelines(str(line) for line in new_top)


def run_prep(out_dir, sys, lig, dif_size=False):
    # Copy the apo pdb
    try:
        subprocess.call(['cp', f"{PREP_INPUTS}/prep.sh", out_dir])
        subprocess.call(['cp', f"{PREP_INPUTS}/prep.mdp", out_dir])
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Define number of Protein+X group
    # group_N = 17 if sys == 'a2b1' and lig in ['A769', 'PF739', 'MT47', 'MK87'] else 22
    # Uncharged system = 17, charged system = 22, with added ions = 24
    group_N = 24
    # Make_ndx command with custom number
    new_line = f'echo -e "name {group_N} Protein_LIG \\n q" | $GMX make_ndx -f {sys}+{lig}.gro -n i.ndx -o i.ndx'
    # Add new line to prep.sh
    with open(f"{out_dir}/prep.sh", 'r') as f:
        lines = f.readlines()
    # Change box min. distance assignment for a2b1 complexes
    if dif_size and sys == 'a2b1':
        for i in np.arange(len(lines)):
            if '-bt dodecahedron' in lines[i]:
                lines[i] = lines[i].replace('1.2', '1.1')
                print('CHANGING BOX SIZE')
    lines.append(new_line)
    with open(f"{out_dir}/prep.sh", 'w') as f:
        f.writelines(lines)
    # Run prep.sh
    try:
        sub = subprocess.Popen(f"cd {out_dir}; bash prep.sh",
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               shell=True
                               )
        out, errors = sub.communicate()
    except:
        print('ERROR')


def fix_itp_includes(out_dir, sys,):
    for s in ['', '2']:
        with open(f"{out_dir}/{sys}_Protein{s}.itp", 'r') as f:
            lines = f.readlines()
        for i in np.arange(len(lines)):
            if all(x in lines[i] for x in ['include', '/home/rhys']):
                lines[i] = f'#include "./{lines[i].split("/")[-1]}'
        with open(f"{out_dir}/{sys}_Protein{s}.itp", 'w+') as f:
            f.writelines(lines)


def setup_minim(dd, sys, lig):
    try:
        subprocess.call(['mkdir', "-p", f"{dd}/01-Min"])
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    try:
        subprocess.call(f"cp {SCRIPT_DIR}/01-Min/* {dd}/01-Min/", shell=True)
        subprocess.call(['cp', f"{dd}/00-Prep/{sys}+{lig}.top", f"{dd}/01-Min/"])
        subprocess.call(['cp', f"{dd}/00-Prep/{sys}+{lig}.gro", f"{dd}/01-Min/"])
        subprocess.call(['cp', f"{dd}/00-Prep/i.ndx", f"{dd}/01-Min/"])
        subprocess.call(['cp', '-r', '/home/rhys/AMPK/Metad_Simulations/System_Setup/force_field_S2P/amber14sb_gmx_s2p.ff', f"{dd}/01-Min/"])
        subprocess.call(f"cp {dd}/00-Prep/*.itp {dd}/01-Min/", shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    try:
        subprocess.call(f"rsync -avzhPu {dd}/01-Min {REMOTE}/{sys}+{lig}/",
                        shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def next_step(ndir):
    for sys in SYSTS:
        for lig in LIGS:
            try:
                subprocess.call(f"rsync -avzhPu {SCRIPT_DIR}/{ndir} {REMOTE}/{sys}+{lig}/", shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))


def make_plumed(source_dat, ref_pdb, out_dat):
    # Extract the atom numbers for input into plumed file...
    with open(ref_pdb, 'r') as f:
        pdb = f.readlines()
    # Read in just atom number (1), res name (3) and res id (5)
    lines = [ln.split() for ln in pdb if 'ATOM' in ln]
    lines = [[int(ln[1]), ln[3], int(ln[5])] for ln in lines]
    # Find those atoms that correspind to the ligand
    ligID = 369 if 'a2b1' in ref_pdb else 368
    lig_atoms = [ln[0] for ln in lines if ln[2] == ligID]
    # Set extent of ligand
    ligN = [min(lig_atoms), max(lig_atoms)]
    # Set extent of protein (assuming from 1 to ligand)
    protN = [1, ligN[0]-1]
    # p0 is same atoms for both a2b1 and a2b2
    p0 = [1334, 166]
    # p1 is different for a2b1 and a2b2
    p1 = [4863, 662] if 'a2b1' in ref_pdb else [4885, 662]

    # Generic lines for readability
    header1 = '#####################################\n#plumed.dat for Funnel MetaD#\n#####################################\n'
    restart = '#RESTART'
    header2 = '\n\n\n###############################################\n###DEFINE RADIUS + CALC PROT-LIG VECTOR COMP###\n###############################################\n'
    header3 = '\n\n\n########################\n###DEFINITION_OF_COMs###\n########################\n'

    # WholeMolecules line that seperates ligand and protein into 2 entities
    WHMline = f"WHOLEMOLECULES STRIDE=1 ENTITY0={protN[0]}-{protN[1]} ENTITY1={ligN[0]}-{ligN[1]}"
    # Ligand atoms
    LIGline = f"lig: COM ATOMS={ligN[0]}-{ligN[1]}"
    # Funnel anchor points
    P_0line = f"\np0: COM ATOMS={p0[0]},{p0[1]}"
    P_1line = f"\np1: COM ATOMS={p1[0]},{p1[1]}\n\n"

    # Write the new plumed.dat file...
    with open(source_dat, 'r') as f:
        lines = f.readlines()
    lines[:0] = [header1, restart, header2, WHMline,
                 header3, LIGline, P_0line, P_1line]
    with open(out_dat, 'w+') as f:
        f.writelines(lines)


def split_pdb(init_pdb):
    pdb = pt.load(init_pdb)
    ligN = pdb.top.n_residues
    pt.write_traj(f'{OUT_DIR}/protein.pdb',
                  pdb[f':1-{ligN-1}'], overwrite=True)
    pt.write_traj(f'{OUT_DIR}/ligand.pdb',
                  pdb[f':{ligN}'], overwrite=True)


if __name__ == "__main__":

    print('started')
    for system in SYSTS:
        for lig in LIGS:
            # wd = f"{OUT_DIR}/{system}+{lig}/00-Prep"
            wd = f"{OUT_DIR}/{system}+{lig}/06-MetaD"
            '''
            print('running')
            make_dirs(wd, system, lig)
            order_check = check_atom_order(f"{wd}/{lig}_4_{system}_4_gmx.pdb",
                                           f"{PARM_DIR}/{lig}.prep")
            assert order_check is True, 'Atoms in PDB not ordered correctly.'
            run_tleap(lig, f"{wd}/{lig}_4_{system}_4_gmx.pdb")
            run_parmed(f"{wd}/{lig}_amb.top",
                       f"{wd}/{lig}_amb.crd",
                       f"{wd}/{lig}")
            run_pdb2gmx(f"{wd}/{system}_apo.pdb", f"{wd}/{system}")
            combine_top(f"{wd}/{system}.top", f"{wd}/{lig}.top")
            combine_gro(f"{wd}/{system}.gro", f"{wd}/{lig}.gro")
            run_prep(wd, system, lig, dif_size=False)
            fix_itp_includes(wd, system)
            setup_minim(f"{OUT_DIR}/{system}+{lig}", system, lig)

            next_step('02-NVT')
            next_step('0345-EQ-MD')



            source_dat = '/home/rhys/AMPK/Metad_Simulations/System_Setup/metad_files/blank_metad.dat'

            try:
                subprocess.call(['mkdir', "-p", wd])
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode, '. Output:', error.output.decode("utf-8"))

            try:
                subprocess.call(f"cp -r {SCRIPT_DIR}/06-MetaD/* {wd}/", shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))

            make_plumed(source_dat, f"{OUT_DIR}/{system}+{lig}/0345-EQ-MD/{system}+{lig}_lastframe.pdb",
                        f"{wd}/plumed_{system}+{lig}.dat")

            try:
                subprocess.call(f"rsync -avzhPu {wd} {REMOTE}/R2/{system}+{lig}/", shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))
            try:
                subprocess.call(f"rsync -avzhPu {wd} {REMOTE}/R3/{system}+{lig}/", shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))
'''
            try:
                subprocess.call(f"rsync -avzhPu {wd} {REMOTE}/R4/{system}+{lig}/", shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))
