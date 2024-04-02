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
import sys

from ..tools import format_pdb


def make_dirs(name, sys, lig):
    # Make the working directory
    try:
        subprocess.run(['mkdir', "-p", name], check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Copy in the apo pdb
    try:
        subprocess.run(['cp',
                        ('/home/rhys/AMPK/Metad_Simulations/System_Setup/'
                         f"apo/{sys}_apo.pdb"),
                        name], check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Copy in the ligand pdb
    try:
        subprocess.run(['cp',
                        ('/home/rhys/AMPK/Metad_Simulations/System_Setup/'
                         f"ligand_pdbs_4_gmx/{lig}_4_{sys}_4_gmx.pdb"),
                        name], check=True)
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


def run_tleap(wd, lig, pdb_path, PARM_DIR):

    lig_params = f"{PARM_DIR}/{lig}"

    ffi_lines = 'source leaprc.gaff2'
    lig_lines = (f"loadamberparams {lig_params}.frcmod\n"
                 f"loadamberprep {lig_params}.prep")
    pdb_lines = f"struc = loadpdb {pdb_path}"
    out_lines = (f"saveamberparm struc {wd}/{lig}_amb.top {wd}/{lig}_amb.crd\n"
                 f"savepdb struc {wd}/{lig}_leap.pdb")

    with open('./temp.tleap', 'w+') as f:
        f.write('\n'.join([ffi_lines, lig_lines,
                           pdb_lines, out_lines, 'quit']))
    # TODO -> remove file
    try:
        subprocess.run(['rm', './leap.log'], check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

    try:
        subprocess.run(['tleap', '-f', './temp.tleap'], check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def run_parmed(prmtop, crd, out_path):
    amber = pmd.load_file(prmtop, crd)
    amber.save(f"{out_path}.top", overwrite=True)
    amber.save(f"{out_path}.gro", overwrite=True)


def run_pdb2gmx(pdb_file, out_name):
    try:
        subprocess.run(["gmx_mpi", "pdb2gmx",
                        "-f", pdb_file,
                        "-o", f"{out_name}.gro",
                        "-p", f"{out_name}.top",
                        "-i", f"{out_name}_posre.itp",
                        "-ff", "amber14sb_gmx_s2p",
                        "-water", "tip3p",
                        "-ignh"],
                       check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def combine_gro(prot_path, lig_path, out_path=None):
    # Load the protein gro file
    with open(prot_path, 'r') as f:
        protein_gro = f.readlines()
    # Load the ligand gro file
    with open(lig_path, 'r') as f:
        ligand_gro = f.readlines()
    # Find the total number of atoms that the new gro will have
    total = (int(ligand_gro[1].split('\n')[0])
             + int(protein_gro[1].split('\n')[0]))
    print(total)
    # Save that total as the start of the document
    line2 = ' '+str(total)+'\n'
    # Combine the the new total, the protein and ligand gro files
    both_gro = chain(protein_gro[0],
                     line2,
                     protein_gro[2:-1],
                     ligand_gro[2:-1],
                     protein_gro[-1])
    # Write the combined file to a new or custom path
    op = (f"{'/'.join(lig_path.split('/')[:-1])}/"
          f"{(prot_path.split('/')[-1][:-4])}"
          f"+{(lig_path.split('/')[-1][:-4])}_built.gro")
    out_name = out_path if out_path else op
    with open(out_name, 'w+') as f:
        f.writelines(str(line) for line in both_gro)


def combine_top(prot_top, lig_top, lig_name=None, directory=None):
    # Set the output path if not pre-defined
    out = directory if directory else f"{'/'.join(lig_top.split('/')[:-1])}"
    # Load the protein .top file
    with open(prot_top) as f:
        pro = f.readlines()
    # Load the ligand .top file
    with open(lig_top) as f:
        d = '['
        lig = [d+e for e in f.read().split(d) if e]
    # Put ligand atomtypes in a separate .itp file
    with open(f"{out}/ligand_atomtypes.itp", 'w') as f:
        f.write([st for st in lig if 'atomtypes' in st][0])
    # Remove all necessary sections from ligand topology
    remove = ['defaults', 'atomtypes', 'molecules', 'system']
    include = '\n'.join([st for st in lig[1:]
                         if not any(x in st for x in remove)])
    # Write the remainder to the ligand itp file
    with open(f"{out}/ligand.itp", 'w+') as f:
        f.write(include)
    # Extract the ligand residue name from the ligand topology
    lig_res_name = lig_name if lig_name else [
        st for st in lig if 'moleculetype' in st][0].split('\n')[2].split()[0]
    print(f"Using Ligand: {lig_res_name}")
    # Define new lines to add to prot topology, with include lines and comments
    include1 = ('; Include ligand atom types \n'
                '#include \"./ligand_atomtypes.itp\" \n\n')
    include2 = '; Include ligand topology   \n#include \"./ligand.itp\" \n\n'
    # Also a line for the very bottom of the file
    # to add to the [molecules] entry
    include3 = f'{lig_res_name}                 1\n'
    # Atomtypes must appear before any [moleculetype] entry
    #    in this case in the chain topologies (if there are multiple protein
    #    chains in the topology
    # n1 = [i for i, s in enumerate(pro) if 'chain' in s][0]
    #    in the case of 1 chain:
    n1 = [i for i, s in enumerate(pro) if 'moleculetype' in s][0]
    # Rest of lig. topology then goes before the water topology is loaded
    n2 = [i for i, s in enumerate(pro) if 'water topology' in s][0]
    # Put the new lines in the correct place in the file
    new_top = chain(pro[:n1], include1,
                    pro[n1:n2], include2,
                    pro[n2:], include3)
    # Write the combined file to a new or custom path
    out_name = (f"{out}/{(prot_top.split('/')[-1][:-4])}"
                f"+{(lig_top.split('/')[-1][:-4])}.top")
    with open(out_name, 'w+') as f:
        f.writelines(str(line) for line in new_top)


def run_prep(out_dir, sys, lig=None, dif_size=False, ngroup=None):
    # AMPK ngroup = 20

    # TODO -> copy files
    # Copy the input scripts
    try:
        subprocess.run(['cp', f"{PREP_INPUTS}/prep_apo.sh", out_dir],
                       check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    try:
        subprocess.run(['cp', f"{PREP_INPUTS}/prep.mdp", out_dir], check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Define number of Protein+X group
    # group_N = 17 if sys == 'a2b1'
    # and lig in ['A769', 'PF739', 'MT47', 'MK87'] else 22
    # Uncharged system = 17, charged system = 22, with added ions = 24
    # Multiple chains: group_N = 24
    if ngroup:
        # Make_ndx command with custom number
        new_line = (f'echo -e "name {ngroup} Protein_LIG \\n q" '
                    f'| $GMX make_ndx -f {sys}+{lig}.gro -n i.ndx -o i.ndx')
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
    try:
        subprocess.run(['bash', 'prep_apo.sh'],
                       cwd=f"{out_dir}",
                       check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def fix_itp_includes(out_dir, sys, numbered=False):
    if numbered:
        for s in ['', '2']:
            with open(f"{out_dir}/{sys}_Protein{s}.itp", 'r') as f:
                lines = f.readlines()
            for i in np.arange(len(lines)):
                if all(x in lines[i] for x in ['include', '/home/rhys']):
                    lines[i] = f'#include "./{lines[i].split("/")[-1]}'
            with open(f"{out_dir}/{sys}_Protein{s}.itp", 'w+') as f:
                f.writelines(lines)
    else:
        for s in ['A', 'B']:
            with open(f"{out_dir}/{sys}_Protein_chain_{s}.itp", 'r') as f:
                lines = f.readlines()
            for i in np.arange(len(lines)):
                if all(x in lines[i] for x in ['include', '/home/rhys']):
                    lines[i] = f'#include "./{lines[i].split("/")[-1]}'
            with open(f"{out_dir}/{sys}_Protein_chain_{s}.itp", 'w+') as f:
                f.writelines(lines)


def setup_minim(dd, sys, lig=None, REMOTE=None):
    # TODO -> Make Dir
    try:
        subprocess.run(['mkdir', "-p", f"{dd}/01-Min"], check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    scripts = ('/home/rhys/phd_tools/simulation_files/'
               'submission_scripts/Local_Dirs')
    # TODO -> Copy Files
    try:
        subprocess.run(f"cp {scripts}/01-Min/* {dd}/01-Min/",
                       shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

    name = f"{sys}+{lig}" if lig else sys
    files = [f"{name}.top",
             f"{name}.gro",
             'i.ndx',
             '*.itp']
    for fn in files:
        # TODO -> Copy Files
        try:
            subprocess.run(f"cp {dd}/00-Prep/{fn} {dd}/01-Min/",
                           check=True,
                           shell=True)
        except subprocess.CalledProcessError as error:
            print('Error code:', error.returncode,
                  '. Output:', error.output.decode("utf-8"))
    # TODO -> Copy Files
    try:
        subprocess.run(('cp -r '
                        '/home/rhys/phd_tools/'
                        'simulation_files/forcefields/amber14sb_gmx_s2p.ff '
                        f"{dd}/01-Min/"),
                       check=True, shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    if REMOTE:
        try:
            subprocess.run(["rsync", "-avzhPu",
                            f"{dd}/01-Min",
                            f"{REMOTE}/{sys}+{lig}/"],
                           check=True)
        except subprocess.CalledProcessError as error:
            print('Error code:', error.returncode,
                  '. Output:', error.output.decode("utf-8"))


def next_step(ndir):
    for sys in SYSTS:
        for lig in LIGS:
            try:
                subprocess.run(["rsync", "-avzhPu",
                                f"{SCRIPT_DIR}/{ndir}",
                                f"{REMOTE}/{sys}+{lig}/"],
                               check=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))


def make_plumed(source_dat, ref_pdb, out_dat,
                ligID, p0, p1, wrap_centre=None):
    '''
    INPUTS:
        source_dat:  Path of template plumed.dat file
        ref_pdb:     Path to system pdb from prev. simulations
        out_dat:     Path to save the new plumed.dat file
        ligID:       Residue number of the ligand in ref_pdb
        p0 & p1:     List of atom indices that define the funnel anchor points
        wrap_centre: The atom id for the
    '''
    # Extract the atom numbers for input into plumed file...
    with open(ref_pdb, 'r') as f:
        pdb = f.readlines()
    # Read only lines containing Atom entries
    lines = [ln.split() for ln in pdb if 'ATOM' in ln]
    # Filter for chain ID:
    resID_col = 4 if lines[0][4].isdigit() else 5
    # Read in just atom number (1), res name (3) and res id (5)
    lines = [[int(ln[1]), ln[3], int(ln[resID_col])] for ln in lines]
    # Find those atoms that correspind to the ligand
    #   N.B. solvent IDs can loop from 9999 to 0, so have to also filter out
    #   any water lines!
    # ligID = 369 if 'a2b1' in ref_pdb else 368
    lig_atoms = [ln[0] for ln in lines if ln[2] == ligID and 'WAT' not in ln]
    # Set extent of ligand
    ligN = [min(lig_atoms), max(lig_atoms)]
    # Set extent of protein (assuming from 1 to ligand)
    protN = [1, ligN[0]-1]
    # p0 is same atoms for both a2b1 and a2b2
    # p0 = [1334, 166]
    # p1 is different for a2b1 and a2b2
    # p1 = [4863, 662] if 'a2b1' in ref_pdb else [4885, 662]

    # Generic lines for readability
    header1 = ('#####################################\n'
               '#    plumed.dat for Funnel MetaD    #\n'
               '#####################################\n')
    restart = '#RESTART'
    header2 = ('\n\n\n###############################################\n'
               '#  DEFINE RADIUS + CALC PROT-LIG VECTOR COMP  #\n'
               '###############################################\n')
    header3 = ('\n\n\n##########################\n'
               '#   DEFINITION_OF_COMs   #\n'
               '##########################\n')
    # WholeMolecules line that seperates ligand and protein into 2 entities
    WHMline = ('\nWHOLEMOLECULES STRIDE=1 ENTITY0=protein ENTITY1=ligand\n')
    # Ligand Centre of Mass
    LIGline = 'lig: COM ATOMS=ligand'

    # Define protein atoms
    protein = (f"\nprotein: GROUP ATOMS={protN[0]}-{protN[1]}\n")
    # Define ligand atoms
    ligand = (f"ligand: GROUP ATOMS={ligN[0]}-{ligN[1]}\n")

    # Funnel anchor points
    P_0line = f"\np0: COM ATOMS={','.join([str(x) for x in p0])}"
    P_1line = f"\np1: COM ATOMS={','.join([str(x) for x in p1])}\n\n"

    # Load the template plumed.dat file...
    with open(source_dat, 'r') as f:
        lines = f.readlines()
    if wrap_centre:
        MOLINFO = f"MOLINFO STRUCTURE=./{ref_pdb.split('/')[-1]}\n"
        WRAPline = (f"\nWRAPAROUND ATOMS=ligand AROUND={wrap_centre} "
                    f"GROUPBY={ligN[1]-ligN[0]+1}")
        lines[:0] =  [header1, restart, header2, MOLINFO, protein, ligand,
                      WHMline, WRAPline, header3, LIGline, P_0line, P_1line]
    else:
        lines[:0] = [header1, restart, header2, protein, ligand,
                    WHMline, header3, LIGline, P_0line, P_1line]
    # Write the new plumed.dat file...
    with open(out_dat, 'w+') as f:
        f.writelines(lines)


def split_pdb(init_pdb):
    pdb = pt.load(init_pdb)
    ligN = pdb.top.n_residues
    pt.write_traj(f'{OUT_DIR}/protein.pdb',
                  pdb[f':1-{ligN-1}'], overwrite=True)
    pt.write_traj(f'{OUT_DIR}/ligand.pdb',
                  pdb[f':{ligN}'], overwrite=True)


def make_readable_gro(gro_path, top_path, tpr=None, out_path=None,
                      reconstruct=False):
    # Set flags for removal of file (if necessary)
    rm_tpr = False
    rm_gro = False
    # If no tpr is provided, use a temporary copy of the prep.mdp
    if not tpr:
        # Path for dummy .mdp file
        tmp_mdp = f"{PREP_INPUTS}/prep.mdp"
        # Temporary file in temp dir.
        tmp_tpr = '/tmp/tmp.tpr'
        # run grompp to create tmp.tpr
        try:
            subprocess.run(["gmx_mpi", "grompp",
                            "-f", tmp_mdp,
                            "-c", gro_path,
                            "-p", top_path,
                            "-o", tmp_tpr],
                           check=True)
        except subprocess.CalledProcessError as error:
            print('Error code:', error.returncode,
                  '. Output:', error.output.decode("utf-8"))
        # Set the temporary tpr for use in trjconv
        tpr = tmp_tpr
        # Flag for the removal of the temp tpr once complete
        rm_tpr = True

    # If no out_path is provided, add Readable_ to the gro file name
    if not out_path:
        out_path = (f"{'/'.join(gro_path.split('/')[:-1])}/"
                    f"Readable_{gro_path.split('/')[-1]}")

    # TODO -> TEST whether echo works without shell!
    # If the -center option is not enough, use brute force
    if reconstruct:
        # Step 1: -pbc whole, produces tmp1.gro
        try:
            subprocess.run(("echo System | gmx_mpi trjconv "
                            f"-f {gro_path} "
                            f"-s {tpr} "
                            "-o /tmp/tmp1.gro "
                            '-pbc whole '),
                           check=True,
                           shell=True)
        except subprocess.CalledProcessError as error:
            print('Error code:', error.returncode,
                  '. Output:', error.output.decode("utf-8"))
        # Step 2: cluster, produces tmp2.gro
        try:
            subprocess.run(("echo Protein System | gmx_mpi trjconv "
                            "-f /tmp/tmp1.gro "
                            f"-s {tpr} "
                            "-o /tmp/tmp2.gro "
                            '-pbc cluster '),
                           check=True,
                           shell=True)
        except subprocess.CalledProcessError as error:
            print('Error code:', error.returncode,
                  '. Output:', error.output.decode("utf-8"))
        # Set the new, clustered gro file as the input for Step 3
        gro_path = '/tmp/tmp2.gro'
        # Flag for the removal of the temp gro files once complete
        rm_gro = True

    # run trjconv to produce a readable output
    try:
        subprocess.run(("echo Protein System | gmx_mpi trjconv "
                        f"-f {gro_path} "
                        f"-s {tpr} "
                        f"-o {out_path} "
                        '-pbc mol -ur compact -center'),
                       check=True,
                       shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # TODO -> Remove files
    # Remove temp tpr if necessary
    if rm_tpr:
        subprocess.run(f"rm {tmp_tpr}", shell=True)
    # TODO -> Remove files
    # Remove temp gro files if necessary
    if rm_gro:
        subprocess.run("rm /tmp/*.gro", shell=True)


def assign_weights(pdb, ligand, out_pdb=None):
    '''
    Edit the final two columns of a .pdb to work with Plumed RMSD and
    other metrics. Final two columns = occupancy and beta factor, but are
    interpreted as alignment weights and displacement measurements weights.

    For ligand RMSD, sets ligand values to 0 and 1 respectively,
    but all proteins atoms to 1 and 0.
    '''
    with open(pdb, 'r') as f:
        lines = f.readlines()
    print(f'Loaded {pdb}')

    # Search for chain ID column in input pdb 
    chain = not [l for l in lines if 'ATOM' in l][0].split()[4].isdigit()
    if chain:
        print('INFO: chain ID detected in pdb')

#     rem = 0
#     while lines[0][-1] not in ['1.00', '0.00']:
#         lines = [l[:-2] for l in lines]
#         rem += 1
#     print(f'Removed {rem} columns from input pdb.')

    for i in np.arange(len(lines)):
        # Discount END lines
        if any(x in lines[i] for x in ['END']):
            continue
        # Remove comment and solvent lines
        if any(x in lines[i] for x in ['REMARK', 'TITLE', 'TER',
                                       'CRYST', 'MODEL', 'WAT']):
            lines[i] = 'DELETORiOUS'
        else:
            l = lines[i].split()
            #print(l)
            if l[3] == ligand:
                if l[-1] == 'H':
                    l[0] = 'DELETORiOUS'
                else:
                    l[-3] = '0.00'
                    l[-2] = '1.00'
            elif any(x == l[2] for x in ['C', 'O', 'N', 'CA']):
                l[-3] = '1.00'
                l[-2] = '0.00'
            else:
                l[0] = 'DELETORiOUS'

            #lines[i] = f'{l[0]:6}{l[1]:>5} {l[2]:^4} {l[3]:3} {l[4]} {l[5]:>3}    {l[6]:>8}{l[7]:>8}{l[8]:>8}{l[9]:>6}{l[10]:>6}\n' 
            lines[i] = format_pdb(l, chain=chain)

    out_file = pdb[:-4]+'_weights.pdb' if not out_pdb else out_pdb
    print(f"WRITING  {out_file}")
    with open(out_file, 'w') as f:
        f.writelines([l for l in lines if 'DELETORiOUS' not in l])


def bb_weights(pdb, ligand, out_pdb=None):
    '''
    '''
    with open(pdb, 'r') as f:
        lines = f.readlines()
    print(f'Loaded {pdb}')

    # Search for chain ID column in input pdb 
    chain = not [l for l in lines if 'ATOM' in l][0].split()[4].isdigit()
    if chain:
        print('INFO: chain ID detected in pdb')

    for i in np.arange(len(lines)):
        # Discount END lines
        if any(x in lines[i] for x in ['END']):
            continue
        # Remove comment and solvent lines
        if any(x in lines[i] for x in ['REMARK', 'TITLE', 'TER',
                                       'CRYST', 'MODEL', 'WAT']):
            lines[i] = 'DELETORiOUS'
        else:
            l = lines[i].split()
            # also remove ligand
            if l[3] == ligand:
                l[0] = 'DELETORiOUS'
            # set backbone atoms to align and measure
            elif any(x == l[2] for x in ['C', 'O', 'N', 'CA']):
                l[-3] = '1.00'
                l[-2] = '1.00'
            else:
                l[0] = 'DELETORiOUS'
            # format output
            lines[i] = format_pdb(l, chain=chain)

    out_file = pdb[:-4]+'_bb.pdb' if not out_pdb else out_pdb
    print(f"WRITING  {out_file}")
    with open(out_file, 'w') as f:
        f.writelines([x for x in lines if 'DELETORiOUS' not in x])


def main():
    PREP_INPUTS = ('/home/rhys/phd_tools/simulation_files/'
                   'submission_scripts/Local_Dirs/00-Prep')

    OUT_DIR = '/home/rhys/Storage/ampk_metad_all_data'
    #PARM_DIR = '/home/rhys/AMPK/Metad_Simulations/System_Setup/ligand_parms'
    SCRIPT_DIR = ('/home/rhys/phd_tools/simulation_files/'
                  'submission_scripts/MareNostrum/class_a')
    REMOTE = 'mn:/home/ub183/ub183944/scratch/ampk_replicas'

    SYSTS = ['a2b1', 'a2b2']
    LIGS = ['A769']
    # LIGS = ['SC4', 'PF739', 'MT47', 'MK87']
    print('started')
    for system in SYSTS:
        for lig in LIGS:
            # wd = f"{OUT_DIR}/{system}+{lig}/00-Prep"
            wd = f"{OUT_DIR}/{system}+{lig}/06-MetaD"

            # This is to set up the system and run uMD
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
            '''

            # This section is to set up the MetaD
            '''
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
            '''

            # This section runs multiple replicas for metaD
            '''
            try:
                subprocess.run(("rsync -avzhPu"
                                f"{wd}"
                                f"{REMOTE}/R2/{system}+{lig}/"),
                               check=True,
                               shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))
            try:
                subprocess.run(("rsync -avzhPu"
                                f"{wd}"
                                f"{REMOTE}/R3/{system}+{lig}/"),
                               check=True,
                               shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))

            try:
                subprocess.run(("rsync -avzhPu"
                                f"{wd}"
                                f"{REMOTE}/R4/{system}+{lig}/"),
                               check=True,
                               shell=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))
            '''


if __name__ == "__main__":
    main()

