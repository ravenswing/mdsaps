import pandas as pd
import matplotlib.pyplot as plt
import pytraj as pt
import subprocess
import sys
import numpy as np
from glob import glob

sys.path.append('/home/rhys/phd_tools/SAPS/')
import traj_tools as tt

systems = ['ship1', 'ship2']

# 
# 'Experimental': ['ship1_R1_A3', 'ship2_R1_D3'],
# POCKETS = {'Tunnel-Front': ['ship1_R2_E1', 'ship2_R3_F2'],
           # 'Tunnel-Back':  ['ship1_R1_C1', 'ship2_R1_A1']}

POCKETS = {'Active-Site': ['ship2_Active-Site+Lig21'],
           'Tunnel-Front': ['ship2_Tunnel-Front+Lig21'],
           'Tunnel-Back':  ['ship2_Tunnel-Back+Lig21']}

# DATA_DIR = '/media/rhys/Storage/ship_holo_uMD/data/'
DATA_DIR = '/media/rhys/Storage/SHIP/Lig21_holo_uMD/data'

# Remote directory: IQTC
SVR_DIR = 'iqtc:/home/g19torces/rhys_running/ship_lig21'

# Run rsync
'''
try:
    subprocess.run(' '.join(['rsync -avzhPu',
                             f"{SVR_DIR}/*",
                             f"{DATA_DIR}/"]),
                   shell=True, check=True)
except subprocess.CalledProcessError as error:
    print('Error code:', error.returncode,
          '. Output:', error.output.decode("utf-8"))

'''


def func(x):
    return int(x.split('ns')[0].split('_')[-1])


# load the rmsd file
BB_RMSD_FILE = '/home/rhys/SHIP/Data/lig21_backbone.h5'
LG_RMSD_FILE = '/home/rhys/SHIP/Data/lig21_ligand.h5'
'''
df = pd.DataFrame(columns=['pocket', 'system', 'data', 'mean', 'std'])
df.to_hdf(BB_RMSD_FILE, key='df', mode='w')
df.to_hdf(LG_RMSD_FILE, key='df', mode='w')
df = pd.read_hdf(BB_RMSD_FILE, key='df', mode='r')
align = '@CA,C,N,O'
'''
# Run analysis
for pocket in POCKETS.keys():
    for system in POCKETS[pocket]:

        wd = f"{DATA_DIR}/{pocket.lower()}/{system}"
        print(wd)

        tt.make_fulltraj(wd, [wd+f"/{system}.top", wd+f"/{system}.eq_6.r"])
        print('made fulltraj')

        # Make a dry topology using PyTraj
        # a = pt.load_topology(f"{wd}/{system}.top")
        # a.strip(':WAT,Na+,Cl-')
        # a.save(f"{wd}/{system}_dry.top")
        # print('made dry top')

        # Identify the most recent traj. file
        traj_to_use = sorted(glob(f"{wd}/*.nc"), reverse=True, key=func)[0]
        print(f"Making analysis with {traj_to_use}")

        # Generate snapshots for PyMol session

        tt.snapshot_pdbs(wd,
                         traj_to_use,
                         f"{wd}/{system}_dry.top",
                         [f"{wd}/{system}.top", f"{wd}/{system}.eq_6.r"],
                         [[0, (50*200)+1, 10*200], [75*200, (100*200)+1, 25*200], [200*200, (1000*200)+1, 100*200]])

        # Original snapshots...
        # [[0, (50*200)+1, 10*200], [75*200, (100*200)+1, 25*200], [200*200, (1000*200)+1, 100*200]])
        print('made snapshots')
        '''

        # Measure backbone RMSD
        trj_path = traj_to_use
        top_path = f"{wd}/{system}_dry.top"
        ref = 0
        print(trj_path)
        print(top_path+'\n')
        rmsd = tt.measure_rmsd(trj_path, top_path, ref,
                               rmsd_mask=align,
                               aln_mask=align,
                               nofit=False)
        rmsd_mean = rmsd.mean()
        rmsd_std = rmsd.std()
        new_ent = pd.DataFrame({'pocket': pocket,
                                'system': system,
                                'data': [rmsd],
                                'mean': rmsd_mean,
                                'std': rmsd_std})
        if df.loc[(df['pocket'] == pocket) & (df['system'] == system)].empty:
            df = pd.concat([df, new_ent], axis=0, ignore_index=True)
        else:
            df.loc[(df['pocket'] == pocket) & (df['system'] == system)] = new_ent.values

# Save backbone RMSD to compressed file
df.to_hdf(BB_RMSD_FILE, key='df', mode='w')


# Load the rmsd file
df = pd.read_hdf(LG_RMSD_FILE, key='df', mode='r')
for pocket in POCKETS.keys():
    for system in POCKETS[pocket]:

        wd = f"{DATA_DIR}/{pocket.lower()}/{system}"

        align = ':1-462@CA,C,N,O' if 'ship1' in system else ':1-457@CA,C,N,O'
        ligN = ':463' if 'ship1' in system else ':458'

        trj_path = sorted(glob(f"{wd}/*.nc"), reverse=True, key=func)[0]
        top_path = f"{DATA_DIR}/{pocket.lower()}/{system}/{system}_dry.top"

        ref = 0
        print(trj_path)
        print(top_path+'\n')
        rmsd = tt.measure_rmsd(trj_path, top_path, ref,
                               rmsd_mask=ligN,
                               aln_mask=align,
                               nofit=True)
        rmsd_mean = rmsd.mean()
        rmsd_std = rmsd.std()
        new_ent = pd.DataFrame({'pocket': pocket,
                                'system': system,
                                'data': [rmsd],
                                'mean': rmsd_mean,
                                'std': rmsd_std})
        if df.loc[(df['pocket'] == pocket) & (df['system'] == system)].empty:
            df = pd.concat([df, new_ent], axis=0, ignore_index=True)
        else:
            df.loc[(df['pocket'] == pocket) & (df['system'] == system)] = new_ent.values

# Save ligand RMSD to compressed file
df.to_hdf(LG_RMSD_FILE, key='df', mode='w')

for metric in ['Ligand', 'Backbone']:

    lr_df = pd.read_hdf(f"/home/rhys/SHIP/Data/lig21_{metric.lower()}.h5", key='df', mode='r')
    print(lr_df)
    colours = ['#0D92FF', '#002982']

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    i = 0
    for pocket in POCKETS.keys():

        data = pd.Series(lr_df.loc[lr_df['pocket'] == pocket].data.values[0])
        print(data)

        fig.tight_layout(h_pad=4)

        plt.suptitle(f"{metric} RMSD w. Rolling Avg.")
        plt.subplots_adjust(top=0.9)

        ax[i].grid()
        xfill = np.linspace(0, 70, len(data))

        ax[i].plot(xfill,
                   pd.Series(data.rolling(1000, center=True).mean()),
                   color=colours[0],
                   zorder=12)
        ax[i].scatter(xfill, data, color=colours[1], alpha=.1, s=.5)
        ax[i].set_xlim([0, 60])
        ymax = 3.0 if 'Back' in metric else 20.0
        ax[i].set_ylim(ymin=0.5, ymax=ymax)
        ax[i].set_title(f"{pocket}")
        ax[i].set_xlabel('Simulation Time / ns')
        ax[i].set_ylabel(f"{metric} RMSD / "+r'$\AA$')
        i += 1

    fig.savefig(f"/home/rhys/Dropbox/RESEARCH/AA_RHYS/BB_BECK/NEW_LIG_21/SHIP2_{metric}_rmsd.png", dpi=300, bbox_inches='tight')
'''
