"""
===============================================================================
                            AMBER SYSTEM PREPARATION
===============================================================================

        Required inputs:



"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parmed as pmd
import pytraj as pt
from glob import glob
from itertools import chain
import subprocess
import os
import re

import ..tools as tt


# .COM FILE MANIPULATION


# RUNNING CHARGE CALCULATIONS


# RUNNING antechamber

def run_antechamber(wd, name):

    # In this step you obtain the prep file
    try:
        subprocess.run(' '.join(['antechamber',
                                 f"-i {wd}/{name}.esp",
                                 '-fi gesp',
                                 f"-o {wd}/{name}.prep",
                                 '-fo prepc',
                                 '-c resp',
                                 f"-ge {wd}/{name}.esp",
                                 '-at gaff']),
                       check=True, shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

    # In this step you will obtain the parameters file (frcmod).
    try:
        subprocess.run(' '.join(['parmchk2',
                                 f"-i {wd}/{name}.prep",
                                 '-f prepc',
                                 f"-o {wd}/frcmod.{name}"]),
                       check=True, shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

    # Generate a pdb file of your ligand with give you the correct order of
    # the atoms of your ligand as follows
    try:
        subprocess.run(' '.join(['antechamber',
                                 f"-i {wd}/{name}.prep",
                                 '-fi prepc',
                                 f"-o {wd}/{name}_amber.pdb",
                                 '-fo pdb']),
                       check=True, shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def reassign_atoms(pdb, reference_pdb, key):
    # Load complex pdb to edit
    with open(pdb, 'r') as f:
        lines = f.readlines()
    print(f'Loaded {pdb}')

    # Identify index of ligand
    start = lines.index('TER\n')
    # Calculate ligand starting atom & residue no.
    atomN = int(lines[start-1].split()[1])+1
    resN = int(lines[start-1].split()[4])+1

    # Open reference pdb from antechamber
    with open(reference_pdb, 'r') as f:
        ref = f.readlines()
    atom_order = [x.split()[2] for x in ref]
    atom_order = [x for x in atom_order if 'H' not in x or x == 'H10']

    new_lines = lines[:start+1]
    coords = lines[start+1:]

    i = 0
    for atm in atom_order:

        l = coords[(key[atm] - 1)].split()

        info = ['ATOM',
                atomN + i,
                atm,
                'MOL',
                resN,
                l[4:6],
                l[7],
                l[8]]
        new_lines.append(tt.format_pdb(info))

        # new_lines.append(f"{'ATOM':6}{atomN+i:>5} {atm:^4} {'MOL':3}   {resN:>3}    {l[4]:>8}{l[5]:>8}{l[6]:>8}{l[7]:>6}{l[8]:>6}\n")
        i += 1

    new_lines += ['TER\n', 'END']

    with open(pdb.split('.')[0]+'_RNM.pdb', 'w') as f:
        f.writelines(''.join(new_lines))


if __name__ == "main":

    wd = '/home/rhys/Dropbox/RESEARCH/AA_RHYS/BB_BECK/NEW_LIG_21'
    name = 'lig21_ship'
