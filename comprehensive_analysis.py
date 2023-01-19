import numpy as np
import pandas as pd
import pickle
from glob import glob

import MDAnalysis as mda
from MDAnalysis.analysis import diffusionmap, align, rms


DATA_DIR = '/media/rhys/Storage/jctc2_rhys_2022/gpfs_data'
NWDA_DIR = '/media/rhys/Storage/jctc2_rhys_2022/New_data/funnel_fragment_paper'

FIGS_DIR = '/media/rhys/Storage/jctc2_rhys_2022/Figures'


EXP_VALS = {'3U5J': -7.65, '3U5L': -8.45, '4HBV': -6.33, '4LR6': -6.11,
            '4MEQ': -4.91, '4PCI': -6.99, '4UYD': -5.59, '2WI3': -4.71,
            '2XHT': -8.13, '2YK9': -6.95, '3OW6': -9.08, '3K99': -9.85,
            '2WI2': -4.71, '3EKO': -9.14, '1JVP': -7.91, '1JVP_B': -7.91,
            '1PXJ': -7.08, '1WCC': -4.71, '2VTA': -5.09, '2VTH': -5.35,
            '2VTM': -4.09, '3TIY': -6.51}

SYSTEMS = {'BRD4': ['3U5L', '3U5J', '4PCI', '4HBV', '4LR6', '4UYD', '4MEQ'],
           'HSP90': ['3K99', '3EKO', '3OW6', '2XHT', '2YK9', '2WI2', '2WI3'],
           'CDK2': ['1JVP', '1JVP_B', '1PXJ', '3TIY', '2VTH', '2VTA', '1WCC',
                    '2VTM']}

# define the limits of each CV per system for processing bias files -> FES
#             sys -method        proj      ext
CV_LIMITS = {'BRD4-fun-metaD':  [0.8, 3.2, 0.0, 0.80],
             'HSP90-fun-metaD': [0.0, 3.7, 0.0, 1.35],
             'CDK2-fun-metaD':  [0.3, 4.2, 0.0, 1.10],
             'BRD4-fun-RMSD':   [0.8, 3.2, 0.0, 1.25],
             'HSP90-fun-RMSD':  [0.0, 3.7, 0.0, 1.50],
             'CDK2-fun-RMSD':   [0.3, 4.2, 0.0, 2.50]}

# default file names for topology and trajectory
# -- SAME AS DOM --
filenames = {'fun-metaD': ['solute.pdb', 'full_trj_solute.dcd'],
             'fun-RMSD': ['equilibrated.pdb', 'trj.dcd']}

# same rmsd threshold for all systems for bound
# -- SAME AS DOM --
bound_rmsd = 2.5

# differing unbound states, due to length of funnel changing
# -- DOM = no unbound used --
unbound_rmsd = {'BRD4': 15,
                'HSP90': 20,
                'CDK2': 25}

# same rmsd threshold for all systems for bound
# -- SAME AS DOM --
bound_rmsd = 2.5

# differing unbound states, due to length of funnel changing
# -- DOM = no unbound used --
unbound_rmsd = {'BRD4': 15,
                'HSP90': 20,
                'CDK2': 25}


def bound_check(x, bound, unbound):
    """ Establish bound, unbound or in middle """
    # check bounds are correct
    assert len(bound) == 2
    # calculate upper bound limit (avg. + std. dev. for proj)
    threshold = bound[0] + bound[1]
    # Value of 1 = bound
    if x < threshold:
        return 1
    # Value of 2 = un-bound
    elif x > unbound:
        return 2
    # Value of 0 = in the middle
    else:
        return 0


def identify_recross(data, metric, bound, unbound):
    """ Count the number of recrossings """
    # calculate status of ligand position: 1 = bound, 2 = unbound
    data['status'] = data[metric].apply(bound_check, args=([bound, unbound]))
    # remove data without a status i.e. not bound or unbound
    middle_ind = data[data.status == 0].index
    data.drop(middle_ind, inplace=True)
    # calculate differences in status column (diff. of 1 = transition)
    data['diffs'] = data.status.diff()
    # identify transitions
    rx_ind = data[data.diffs != 0].index
    # extract times as list for plotting
    rx = [1.]+[t for t in data.loc[rx_ind[1:]].time.tolist()][1::2]
    # count number of recrossings
    N = int((len(rx)-1))
    # output number of RX and list of RX times
    return N, rx


def calc_rmsd():
    # THIS WILL NOT WORK!!!!! DON´T RUN!
    # NEED TO APPLY PREVIOUS CHANGES

    for method in ['fun-metaD', 'fun-RMSD']:
        for system in SYSTEMS.keys():
            for pdb in SYSTEMS[system]:
                for i in range(3):
                    solute_pdb = (f'{DATA_DIR}/{method}/{system}/{i}_output/'
                                  f'{pdb}/{filenames[method][0]}')
                    solute_traj = (f'{DATA_DIR}/{method}/{system}/{i}_output/'
                                   f'{pdb}/{filenames[method][1]}')
                    u = mda.Universe(solute_pdb, solute_traj)
                    r = rms.RMSD(u, select='backbone',
                                 groupselections=['resname MOL and not name H*'],
                                 ref_frame=0).run()
                    lig_rmsd = r.results.rmsd[:, -1]
                    cache_file = f'{RMSD_DIR}/{method}/{system}/{pdb}_{i}.p'
                    with open(cache_file, 'wb') as f:
                        pickle.dump(lig_rmsd, f)


def find_rx():
    # define delay (ns) - i.e. no RX will be counted within ns from the start
    # -- SAME AS DOM (for fun-metaD) --
    delay = 0

    # create storage df
    data = pd.DataFrame(columns=['method', 'system', 'pdb', 'rep',
                                 'number', 'rx'])

    for method in ['fun-metaD', 'fun-RMSD']:
        if 'RMSD' in method:
            location = ('/home/rhys/Storage/jctc2_rhys_2022/'
                        'Fun-RMSD/analysis/RMSD_Data')
            rep_range = [0, 1, 2]
        else:
            location = NWDA_DIR+'/RMSD_Data'
            rep_range = [3, 4, 5]
        for system in SYSTEMS.keys():
            for pdb in SYSTEMS[system]:
                for i in rep_range:
                    if method == 'fun-metaD' and system == 'BRD4' and i == 5:
                        continue
                    df = pd.DataFrame()
                    with open(f'{location}/{method}/{system}/{pdb}_{i}.p', 'rb') as f:
                        rmsd_data = pickle.load(f)
                    df['RMSD'] = rmsd_data
                    # add time and convert to ns
                    #df['time'] = np.arange(0,len(df)) if 'RMSD' in method else np.arange(0,len(df))*0.1
                    df['time'] = np.arange(0, len(df))
                    # run recrossing counting function
                    N, rx = identify_recross(df, 'RMSD', bound=[bound_rmsd, 0.], unbound=unbound_rmsd[system])
                    # filter out initial values due to the defined delay
                    rx = [1.]+[x for x in rx[1:] if x > delay]
                    # add values to data storage
                    data.loc[len(data.index)] = [method, system, pdb, i, N, rx]
    # save data
    data.to_hdf('/media/rhys/Storage/jctc2_rhys_2022/NEW_rx_from_rmsd.h5', key='df', mode='w')


def _work_out_biasfile(bias_dir, t):
    d = {}
    for bias_file in glob(f'{bias_dir}/*'):
        bias = np.load(bias_file)
        total_bias = np.sum(bias)
        d[bias_file] = total_bias

    sort_d = sorted(d.items(), key=lambda x: x[1])
    sorted_bias_files = [i[0] for i in sort_d]
    N = len(sorted_bias_files)
    print(bias_dir, N)

    t_per_file = np.linspace(0, 2000, N)
    print(t_per_file)
    # bisect.bisect_left(t_per_file, t)  # is faster for a large list than...
    index_to_use = next(i for i, x in enumerate(t_per_file) if x >= t)
    print(index_to_use)
    return sorted_bias_files[index_to_use]


def _write_fes_from_bias(bias_path, system, method, out_path):

    print(f"INFO: Writing FES for {bias_path}")
    # method taken directly from DOM
    deltaT = 300*(10-1)

    # load the bias file        
    bias = np.load(bias_path, allow_pickle=True)
    # define the CV space
    xticks = np.linspace(CV_LIMITS[f"{system}-{method}"][0],
                         CV_LIMITS[f"{system}-{method}"][1], len(bias[0, :]))
    yticks = np.linspace(CV_LIMITS[f"{system}-{method}"][2],
                         CV_LIMITS[f"{system}-{method}"][3], len(bias[:, 0]))

    # fes[y,x]
    with open(out_path, 'w') as f:
        f.write('#proj,cv2,bias(kj/mol)\n')
        for y in range(len(bias[:, 0])):
            for x in range(len(bias[0, :])):
                f.write('%14.9f %14.9f %14.9f\n'% (xticks[x], yticks[y], -((300+deltaT)/deltaT)*bias[y,x]))
    print(f"INFO: Successfully written FES for {out_path}")


def extract_fes_per_rx():
    # Read in the rx information.
    rmsd_data = pd.read_hdf('/media/rhys/Storage/jctc2_rhys_2022/NEW_rx_from_rmsd.h5', key='df')

    for method in ['fun-metaD', 'fun-RMSD']:
        if 'RMSD' in method:
            rep_range = [0, 1, 2]
        else:
            rep_range = [3, 4, 5]
        for system in SYSTEMS.keys():
            for pdb in SYSTEMS[system]:
                for i in rep_range:
                    if method == 'fun-metaD' and system == 'BRD4' and i == 5:
                        continue
                    wd = f"{NWDA_DIR}/{method}/{system}/{i}_output/{i}_output/{pdb}"

                    # Get list of rx from rmsd data.
                    recrossings = rmsd_data.loc[(rmsd_data.method==method) &
                                                (rmsd_data.system==system) &
                                                (rmsd_data.pdb==pdb) &
                                                (rmsd_data.rep==i)].rx.values[0][1:]

                    # 
                    for i, rx in enumerate(recrossings):
                        bias_path = _work_out_biasfile(f"{wd}/bias_dir", rx)

                        print(bias_path)

                        _write_fes_from_bias(bias_path,
                                             system,
                                             method,
                                             f"{wd}/rxFES_{i}_{rx}.dat")

def calculate_delta_g(fes_path, A, B, vol_corr):
    fes_data = pd.read_table(fes_path, sep="\s+", header=0, names=['proj','ext','val'])
    # convert CVs to Angstroms
    fes_data.proj = fes_data.proj.multiply(10)
    fes_data.ext = fes_data.ext.multiply(10)
    # isolate the values that correspond to the basins
    basin_A = fes_data[(fes_data.proj.between(A[0], A[1])) & (fes_data.ext.between(A[2], A[3]))].val
    basin_B = fes_data[(fes_data.proj.between(B[0], B[1])) & (fes_data.ext.between(B[2], B[3]))].val
    # calculate the dG from the minimum value in each basin (bound - unbound)
    delta_g = basin_A.min() - basin_B.min()
    # convert to kcal and apply volume correction
    delta_g = (delta_g / 4.184) + vol_corr
    return delta_g


#                            proj       ext
proj_basins = {'BRD4_B':  [8.0, 13.0, 0.0, 6.0],  # A 1 = Narrow definition
               'BRD4_U':  [25., 30.0, 0.0, 3.0],  # B
               'HSP90_B': [8.0, 12.0, 0.0, 3.0],  # A
               'HSP90_U': [30., 35.0, 0.0, 3.0],  # B
               'CDK2_B':  [5.0, 11.0, 0.0, 2.0],  # A
               'CDK2_U':  [30., 40.0, 0.0, 3.0]}  # B

proj_basins_wide = {'BRD4_B':  [8.0, 18.0, 0.0, 6.0],  # A 2 = Wide definitio
                    'BRD4_U':  [25., 30.0, 0.0, 3.0],  # B
                    'HSP90_B': [8.0, 16.0, 0.0, 8.0],  # A
                    'HSP90_U': [30., 35.0, 0.0, 3.0],  # B
                    'CDK2_B':  [5.0, 16.0, 0.0, 8.0],  # A
                    'CDK2_U':  [30., 40.0, 0.0, 3.0]}  # B

#                          proj       rmsd
rmsd_basins = {'BRD4_B':  [0.0, 15.0, 0.0, 5.0],  # A
               'BRD4_U':  [25., 30.0, 0.0, 12.],  # B
               'HSP90_B': [5.0, 11.0, 0.0, 5.0],  # A
               'HSP90_U': [30., 35.0, 0.0, 14.],  # B
               'CDK2_B':  [8.0, 15.0, 0.0, 5.0],  # A
               'CDK2_U':  [30., 40.0, 0.0, 20.]}  # B

# volume corrections as calculated from Laco's notebook
vol_corr = {'BRD4':  2.901782970733069,
            'HSP90': 3.0003470683770175,
            'CDK2':  2.955066838544255}


def make_tables():
    # print dG values for all systems
    for method in ['fun-metaD', 'fun-RMSD']:
        if 'RMSD' in method:
            rep_range = [0, 1, 2]
            basins = rmsd_basins
        else:
            rep_range = [3, 4, 5]
            basins = proj_basins

        to_csv = ['System,Ligand,Replica,NRx,Rx1,tRx1,Rx2,tRx2,Rx3,tRx3,RxMax,tRxMax,Exp\n']
        for system in SYSTEMS.keys():
            for pdb in SYSTEMS[system]:
                for i in rep_range:
                    new_line = f"{system},{pdb},{i},"
                    if method == 'fun-metaD' and system == 'BRD4' and i == 5:
                        continue
                    wd = f"{NWDA_DIR}/{method}/{system}/{i}_output/{i}_output/{pdb}"
                    nRx = len(glob(f'{wd}/rxFES_*'))
                    new_line += f"{nRx},"
                    nFES = min(len(glob(f"{wd}/rxFES_*")), 3)
                    for n in range(nFES):
                        fes_path = glob(f"{wd}/rxFES_{n}_*")[0]
                        print(fes_path)
                        dg = calculate_delta_g(fes_path,
                                               basins[f"{system}_B"],
                                               basins[f"{system}_U"],
                                               vol_corr[system])
                        new_line += f"{dg:.3f},{fes_path.split('/')[-1].split('_')[-1].split('.')[0]},"
                    new_line += ('-,'*2*max(0,(3-nFES)))
                    if nFES:
                        final_rx = glob(f"{wd}/rxFES_{nRx-1}_*")[0]
                        dg_final = calculate_delta_g(final_rx,
                                                basins[f"{system}_B"],
                                                basins[f"{system}_U"],
                                                vol_corr[system])
                        new_line += f"{dg_final:.3f},{final_rx.split('/')[-1].split('_')[-1].split('.')[0]},"
                    else:
                        new_line += '-,-,'
                    new_line += f"{EXP_VALS[pdb]}"
                    to_csv.append(new_line + '\n')

        with open(f"/media/rhys/Storage/jctc2_rhys_2022/{method.lower()}_dg_values.csv", 'w') as f:
            f.writelines(to_csv)


if __name__ == "__main__":

    # 1 - CALCULATE THE RX
    # calc_rmsd()
    # find_rx()

    # 2, 3, 4 
    # extract_fes_per_rx()

    # 5 - 
    make_tables()
