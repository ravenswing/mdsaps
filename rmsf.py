import numpy as np
import subprocess


def main():
    DATA_DIR = '/home/rhys/Storage/ampk_metad_all_data'
    method = 'fun-metaD'

    systems = {'a2b1': ['A769', 'PF739', 'SC4', 'MT47', 'MK87'],
               'a2b2': ['A769', 'PF739', 'SC4', 'MT47', 'MK87']}
    reps = ['R'+str(x) for x in np.arange(4)+1]

    hdf_path = f"{DATA_DIR}/analysis_data/{method}_rmsf.h5"
    for system in systems:
        for pdb in systems[system]:
            for rep in reps:
                wd = f"{DATA_DIR}/{method}/{system}+{pdb}/{rep}"
                top = f"{wd}/md_dry.pdb"
                traj = f"{wd}/{system}+{pdb}_short.xtc"
                protein = '\"protein or resid 303\"'

                # Measure the RMSF
                try:
                    subprocess.run((f"python measure_rmsf.py {top} {traj} "
                                    f"{hdf_path} backbone {protein} -res -ids "
                                    f"{system} {pdb} {rep}"),
                                   shell=True,
                                   check=True)
                except subprocess.CalledProcessError as error:
                    print('Error code:', error.returncode,
                          '. Output:', error.output.decode("utf-8"))


if __name__ == '__main__':
    main()
