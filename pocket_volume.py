import subprocess
import MDAnalysis as mda
from MDAnalysis.analysis import align

from . import tools as tt


def aligned_pdb(wd: str, ref_path: str) -> None:
    u = tt._init_universe(f"{wd}/md_dry.pdb")
    protein = u.select_atoms("protein or resname S2P")
    with mda.Writer(f'{wd}/tmp_prot.pdb', protein.n_atoms) as W:
        for ts in u.trajectory:
            W.write(protein)

    mobile = tt._init_universe(f'{wd}/tmp_prot.pdb')
    ref = tt._init_universe(ref_path)
    aligner = align.AlignTraj(mobile, ref, select='backbone',
                              filename=f'{wd}/aligned.pdb').run()


def aligned_dcd(wd: str, xtc_name: str, ref_path: str) -> None:
    # ADD NFRAMES ARGUMENT AN LINSPACE FOR FRAME ITERATION!
    u = tt._init_universe([f"{wd}/md_dry.pdb", f"{wd}/{xtc_name}"])
    protein = u.select_atoms("protein or resname S2P")
    with mda.Writer(f'{wd}/tmp_prot.xtc', protein.n_atoms) as W:
        for ts in u.trajectory[::5]:
            W.write(protein)

    mobile = tt._init_universe([f'{wd}/aligned.pdb', f'{wd}/tmp_prot.xtc'])
    ref = tt._init_universe(ref_path)
    aligner = align.AlignTraj(mobile, ref, select='backbone',
                              filename=f'{wd}/aligned.dcd').run()


def pocket_select(wd, out_name):
    mpck_cmd = ["mdpocket",
                "--trajectory_file", "aligned.dcd ",
                "--trajectory_format", "dcd",
                "-f", "aligned.pdb",
                "-o",  f"{out_name}",
                "-n" "3.0"]
    try:
        subprocess.run(mpck_cmd, cwd=wd, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def pocket_volume(wd, out_name, ref_path):
    # TODO -> COPY FILES
    try:
        subprocess.run(f'cp {ref_path} {wd}', shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))

    mpck_cmd = ["mdpocket",
                "--trajectory_file", "aligned.dcd",
                "--trajectory_format", "dcd ",
                "-f", "aligned.pdb",
                "--selected_pocket", f"{ref_path.split('/')[-1]}",
                "-o", out_name,
                "-n", "3.0",
                "-v", "10000"]
    try:
        subprocess.run(mpck_cmd, cwd=wd, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def main():
    systems = {'a2b1': ['A769', 'PF739', 'SC4', 'MT47', 'MK87'],
               'a2b2': ['A769', 'PF739', 'SC4', 'MT47', 'MK87']}

    DATA_DIR = '/home/rhys/Storage/ampk_metad_all_data'
    TMPL_DIR = f"{DATA_DIR}/pockets/cut_templates"

    for method in ['fun-metaD']:
        for system in systems:
            out_dir = f"/media/rhys/Storage/ampk_metad_all_data/analysis_data/mdpocket/{system}"
            ref = f"/home/rhys/Storage/ampk_metad_all_data/super_ref/{system}.pdb"
            for pdb in systems[system]:
                for rep in ['R1', 'R2', 'R3', 'R4']:
                    wd = f"{DATA_DIR}/{method}/{system}+{pdb}/{rep}"
                    """
                    aligned_pdb(wd, ref)
                    aligned_dcd(wd, f"{system}+{pdb}_{rep}_GISMO.xtc", ref)

                    try:
                        subprocess.run('rm tmp_*', cwd=wd, shell=True, check=True)
                    except subprocess.CalledProcessError as error:
                        print('Error code:', error.returncode,
                            '. Output:', error.output.decode("utf-8"))

                    pocket_select(wd, f"{system}+{pdb}_{rep}")

                    # TODO -> COPY FILES
                    try:
                        subprocess.run(f'cp *_freq_iso_* {out_dir}', cwd=wd,
                                    shell=True, check=True)
                    except subprocess.CalledProcessError as error:
                        print('Error code:', error.returncode,
                            '. Output:', error.output.decode("utf-8"))
                    # TODO -> COPY FILES
                    try:
                        subprocess.run(f'cp *_atom_pdens* {out_dir}', cwd=wd,
                                    shell=True, check=True)
                    except subprocess.CalledProcessError as error:
                        print('Error code:', error.returncode,
                            '. Output:', error.output.decode("utf-8"))
                    """

                    pocket_volume(wd, f"{system}+{pdb}_{rep}_vol",
                                  f"{TMPL_DIR}/{system}+{pdb}_cut.pdb")


if __name__ == '__main__':
    main()
