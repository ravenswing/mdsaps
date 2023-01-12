"""
===============================================================================
                        DOCKING WITH AUTODOCK VINA
===============================================================================
"""

import pandas as pd
from glob import glob
import subprocess


VINA_EXEC = '/home/rhys/software/vina_1.2.3_linux_x86_64'


def _read_grid(dir_path):
    """ Read the grid file and convert npts dimension to size """
    # read the file
    with open(glob(f"{dir_path}/grid*")[0], 'r') as f:
        grid_file = f.readlines()
    for line in grid_file:
        # spacing (Angstroms) between grid points
        if 'spacing' in line:
            spacing = float(line.split()[-1])
        # number of points in x,y,z
        elif 'npts' in line:
            dims = list(map(float, line.split()[1:]))
        # center box coordinate x,y,z
        elif 'center' in line:
            centre = list(map(float, line.split()[1:]))
    # calculate box size = npts x spacing
    size = list(map(lambda x: x*spacing, dims))
    # return 2 lists of x,y,z values
    return centre, size


def _run_vina(prot_path, lig_path, centre, size, out_dir):
    """ Run AutoDock Vina  """
    out_fn = (f"{prot_path.split('/')[-2]}/"
              + f"{prot_path.split('/')[-1].split('.')[0]}_out.pdbqt")
    # construct the command
    vina_command = [f"{VINA_EXEC}",
                    '--receptor', prot_path,
                    '--ligand', lig_path,
                    f"--center_x {centre[0]}",
                    f"--center_y {centre[1]}",
                    f"--center_z {centre[2]}",
                    f"--size_x {size[0]}",
                    f"--size_y {size[1]}",
                    f"--size_z {size[2]}",
                    f"--out {out_dir}/{out_fn}"]
    # run vina
    try:
        subprocess.call(' '.join(vina_command), shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def _read_output(out_file):
    """ Read the Vina Output PDBQT file
        Results:
            [0] affinity (kcal/mol)
            dist from best mode:
            [1] rmsd l.b.
            [2] rmsd u.b.
    """
    with open(out_file, 'r') as f:
        # search for data lines and convert to floar
        results = [list(map(float, line.split('\n')[0].split()[3:]))
                   for line in f.readlines() if 'VINA RESULT' in line]
    return results


def process_pocket(pocket_dir, output_dir):
    c, s = _read_grid(pocket_dir)
    for system in SYSTEMS:
        for prot_file in sorted(glob(f"{pocket_dir}/ship*.pdbqt")):

            _run_vina(prot_file, ligand_file, c, s, output_dir)


def process_results(results_dir):

    data = pd.DataFrame(columns=['pocket', 'system', 'conformation', 'pose',
                                 'affinity', 'rmsd_lb', 'rmsd_ub'])

    for vina_output in sorted(glob(f"{results_dir}/*/*out.pdbqt")):
        print(vina_output)
        naming = vina_output.split('/')[-1].split('.')[0].split('_')
        pose_list = _read_output(vina_output)
        i = 1
        for p in pose_list:
            data.loc[len(data.index)] = [naming[3][:-6],
                                         naming[0],
                                         naming[1]+'_'+naming[2],
                                         i,
                                         p[0], p[1], p[2]]
            i += 1
    return data


def create_csv(data, output_dir):
    # create csv
    for pocket in POCKETS:
        to_csv = []
        for system in SYSTEMS:
            df = data.loc[(data.pocket == pocket) & (data.system == system)]
            for conf in pd.unique(df.conformation.values):

                d1 = f"{system},{conf},Pose,"
                d2 = f"{system},{conf},Aff.,"
                d3 = f"{system},{conf},RMSD LB,"
                d4 = f"{system},{conf},RMSD UB,"
                for p in df.loc[(df.conformation == conf)].pose.values:
                    d1 += f"{p},"
                    d2 += f"{df.loc[(df.conformation==conf) & (df.pose==p)].affinity.values[0]},"
                    d3 += f"{df.loc[(df.conformation==conf) & (df.pose==p)].rmsd_lb.values[0]},"
                    d4 += f"{df.loc[(df.conformation==conf) & (df.pose==p)].rmsd_ub.values[0]},"

                to_csv.append(d1 + '\n')
                to_csv.append(d2 + '\n')
                to_csv.append(d3 + '\n')
                to_csv.append(d4 + '\n')
        with open(f"{output_dir}/{pocket.lower()}_vina_output.csv", 'w') as f:
            f.writelines(to_csv)


if __name__ == '__main__':

    DRBX_DIR = '/home/rhys/Dropbox/RESEARCH/AA_RHYS/BB_BECK/Snapshots_4_Docking'

    # POCKETS = ['Experimental', 'Xray', 'Tunnel-Front', 'Tunnel-Back']
    POCKETS = ['Active-Site']

    SYSTEMS = ['ship1', 'ship2']

    # ligand file is the same for all systems
    # ligand_file = f"{DRBX_DIR}/lig_paper_Ship.pdbqt"
    ligand_file = f"{DRBX_DIR}/lig21_ship.pdbqt"

    output_dir = f"{DRBX_DIR}/Lig21_Vina_Results"

    """
    c, s = _read_grid(f"{DRBX_DIR}/Tunnel-Front_Pocket/")
    _run_vina(f"{DRBX_DIR}/Tunnel-Front_Pocket/ship1_R1_C3_Tunnel-FrontPocket.pdbqt",
              ligand_file,
              c, s)
    """

    for pocket in POCKETS:
        wd = f"{DRBX_DIR}/{pocket}_Pocket"
        process_pocket(wd, output_dir)

    POCKETS = ['Experimental', 'Xray', 'Tunnel-Front', 'Tunnel-Back', 'Active-Site']
    new_data = process_results(output_dir)
    create_csv(new_data, output_dir)
