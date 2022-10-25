"""
===============================================================================
                            AMBER SYSTEM PREPARATION
===============================================================================

        Required inputs:
            - PDB of apo protein
            - PDB of ligand
            - Ligand Parameters: .frcmod & .prep
"""


import numpy as np
import parmed as pmd
import pytraj as pt
from itertools import chain
import subprocess




if __name__ == "main":
