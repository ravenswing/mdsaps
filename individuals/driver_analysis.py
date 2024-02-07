import numpy as np
from itertools import product
import subprocess
import sys

sys.path.append('/home/rhys/phd_tools/SAPS')
import traj_tools as tt

DATA_DIR = '/home/rhys/Storage/ampk_metad_all_data'
SAVE_DIR = '/home/rhys/Dropbox/RESEARCH/AA_RHYS/BB_AMPK/Plots'

method = 'fun-metaD'

systems = {'a2b1': ['A769', 'PF739', 'SC4', 'MT47', 'MK87'],
           'a2b2': ['A769', 'PF739', 'SC4', 'MT47', 'MK87']}

reps = ['R'+str(x) for x in np.arange(4)+1]


for system in systems.keys():
    for pdb in systems[system]:
        for rep in reps:
            wd = f"./{system}+{pdb}/{rep}"

            # DRIVER
            try:
                subprocess.run(f"cp  {system}_driver.in {wd}/",
                               shell=True,
                               check=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))
            try:
                subprocess.run(('plumed driver '
                                f"--plumed {wd}/{system}_driver.in "
                                f"--pdb {wd}/md_dry.pdb "
                                f"--mf_xtc {wd}/metad_{system}+{pdb}_final.xtc "
                                '--trajectory-stride 1000 --timestep 0.002'),
                               shell=True,
                               check=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))

            try:
                subprocess.run(f"mv loop_distance.dat {wd}/",
                               shell=True,
                               check=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))

            # RMSF
            '''
            subprocess.run(("echo Backbone | "
                            "gmx_mpi rmsf "
                            f"-s {wd}/prod.tpr "
                            f"-f {wd}/metad_{system}+{pdb}_final.xtc "
                            f"-o {wd}/rmsf.xvg "
                            "-fit yes -res "),
                            shell=True)

            '''
# RMSDs
'''
to_measure = 'backbone and resnum 273-365'
to_align = 'backbone and resnum 1-272'
out_name = 'Beta_AlphaAligned'

sys_info = list(product(*[['a2b1', 'a2b2'],
                            ['A769', 'PF739', 'SC4', 'MT47', 'MK87'],
                            ['R1', 'R2', 'R3', 'R4']]))

tt.calculate_rmsd(sys_info,
                    f"{DATA_DIR}/{method}"+"/{p[0]}+{p[1]}/{p[2]}/md_dry.pdb",
                    f"{DATA_DIR}/{method}"+"/{p[0]}+{p[1]}/{p[2]}/metad_{p[0]}+{p[1]}_final.xtc",
                    f"{DATA_DIR}/analysis_data/Beta_AlphaAligned_rmsd.h5",
                    to_measure,
                    align=to_align,
                    ref_frmt=0,
                    unique_ligs=True)
'''
