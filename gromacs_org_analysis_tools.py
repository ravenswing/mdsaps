"""
===============================================================================
                    Gromacs Organisation n' Analysis Tools
===============================================================================

    - sumhills
"""

import numpy as np
import pandas as pd
import pickle
import subprocess
import sys

sys.path.append('/home/rhys/phd_tools/python_scripts')
import load_data as load

sys.path.append('/home/rhys/phd_tools/SAPS')
import traj_tools as tt


def run_sumhills(wd, out_name, stride=None, cv=None):
    """ Outputs:
        - FES
        - FES over time (with stride)
        - 1D FES (with cv)
    """
    # Create FESs over time is stride is provided
    if stride is not None:
        # Make a new directory to hold output
        subprocess.run(f"mkdir -p {wd}/fes", shell=True, check=True)
        # Adjust output name for new directory
        out_name = f'fes/{out_name}'
        # Add flag for plumed command
        st_flag = f" --stride {stride}"
    else:
        st_flag = ''
    # Create 1D FES if cv is specified
    if cv is not None:
        # Add flag for plumed command (assuming 300K!)
        cv_flag = f"--idw {cv} --kt 2.49"
    else:
        cv_flag = ''
    # Construct plumed command
    cmd = (f"plumed sum_hills --hills {wd}/HILLS "
           f"--outfile {wd}/{out_name}_FES --mintozero {st_flag} {cv_flag}")
    # Execute the plumed sum_hills command
    try:
        subprocess.run(cmd,
                       shell=True,
                       check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def gismo_traj(wd, in_path, out_path, tpr='prod.tpr', ndx='i.ndx'):
    """ cutdown the trajectories using Gromacs trjconv ready for GISMO """
    # call gmx trjconv with -dt 100 to cut down the trajectory
    cmd = ("echo Backbone Protein_LIG | gmx_mpi trjconv "
           f"-s {wd}/{tpr} "
           f"-f {wd}/{in_path} "
           f"-o {wd}/{out_path} "
           f"-n {wd}/{ndx} "
           "-fit rot+trans "
           "-dt 100 ")
    try:
        subprocess.run(cmd,
                       shell=True,
                       check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))


def gismo_colvar(wd, in_colvar='COLVAR', out_colvar='COLVAR_GISMO'):
    """ combine old and reweighted colvars """
    # Load in the original COLVAR
    old_col = load.colvar(f"{wd}/{in_colvar}", 'as_pandas')

    # Cutdown old COLVAR to match trajectories by selecting every 5th line
    old_col = old_col.iloc[::5, :]
    # Add every 10th line (and the second line) for GISMO colvar
    gis_col = old_col.iloc[:2, :]
    gis_col = gis_col.append(old_col.iloc[10::10, :], ignore_index=True)

    # Define path for the output GISMO COLVAR file
    gismo_col_path = f"{wd}/{out_colvar}"
    # Add the header line to this new COLVAR
    with open(gismo_col_path, 'w') as f:
        f.write("#! FIELDS "+" ".join(list(gis_col.columns.values))+"\n")
    # Save the cutdown GISMO COLVAR
    gis_col.to_csv(gismo_col_path, sep=" ",
                   header=False, index=False, mode='a')
    print(f"Successfully converted {in_colvar} to {out_colvar}.")


def _bound_check(x, bound, unbound):
    """ Establish bound, unbound or in middle """
    # Check bounds are correct
    assert len(bound) == 2
    # Calculate upper bound limit (avg. + std. dev. for proj)
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


def _identify_recross(data, metric, bound, unbound):
    """ Count the number of recrossings """
    # Calculate status of ligand position: 1 = bound, 2 = unbound
    data['status'] = data[metric].apply(_bound_check, args=([bound, unbound]))
    # Remove data without a status i.e. not bound or unbound
    middle_ind = data[data.status == 0].index
    data.drop(middle_ind, inplace=True)
    # Calculate differences in status column (diff. of 1 = transition)
    data['diffs'] = data.status.diff()
    # Identify transitions
    rx_ind = data[data.diffs != 0].index
    # Extract times as list for plotting
    rx = [1.]+[t for t in data.loc[rx_ind[1:]].time.tolist()][1::2]
    # Count number of recrossings
    N = int((len(rx)-1))
    # Output number of RX and list of RX times
    return N, rx


def rx():

    # FROM COLVAR:
    # create storage df
    data = pd.DataFrame(columns=['system', 'lig', 'rep', 'number', 'rx'])

    for system in SYSTS:
        for lig in LIGS:
            for rep in ['R'+str(x) for x in np.arange(2)+1]:
                print(system, lig, rep)
                # load the colvar into DataFrame
                df = load.colvar(f"{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/COLVAR", 'as_pandas')
                # rename CV columns (- pp.)
                df.rename(columns={'pp.proj':'proj', 'pp.ext':'ext'}, inplace=True)
                # remove unnecessary columns
                df.drop(columns=[col for col in df if col not in ['time', 'proj', 'ext']], inplace=True)
                # convert time to ns
                df.time = df.time.multiply(0.001)
                # convert CVs to Angstroms
                df.proj = df.proj.multiply(10)
                df.ext = df.ext.multiply(10)
                df['mean'] = df['proj'].rolling(500,center=True).mean()
                # identify bound state from initial projection value
                init_proj = [df['proj'].iloc[0], 1.5]
                print(init_proj[0]+init_proj[1])
                # identify bound state from BASIN?
                #init_proj = [10., 0.]
                # run recrossing counting function
                N, rx = _identify_recross(df, 'proj', bound=init_proj, unbound=25)
                # add values to data storage
                data.loc[len(data.index)] = [system, lig, rep, N, rx]

    # save data
    data.to_hdf(f"{DATA_DIR}/Replicas_rx_from_proj.h5", key='df', mode='w')
 

    # FROM PANDAS:
    # create storage df
    data = pd.DataFrame(columns=['method', 'system', 'pdb', 'rep',
                                 'number', 'rx'])

    for method in ['fun-metaD', 'fun-RMSD']:
        if 'RMSD' in method:
            location = f"{STEM}/Fun-RMSD/analysis/RMSD_Data"
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
                    # df['time'] = np.arange(0,len(df)) if 'RMSD' in method else np.arange(0,len(df))*0.1
                    df['time'] = np.arange(0, len(df))
                    # run recrossing counting function
                    N, rx = _identify_recross(df, 'RMSD',
                                             bound=[bound_rmsd, 0.],
                                             unbound=unbound_rmsd[system])
                    # add values to data storage
                    data.loc[len(data.index)] = [method, system, pdb, i, N, rx]
    # save data
    data.to_hdf(f"{STEM}/NEW_rx_from_rmsd.h5", key='df', mode='w')


def calculate_delta_g(fes_path, CVs, A, B,
                      vol_corr=0):
    fes_data = load.fes(fes_path)
    # Rename CV columns to 1 & 2
    fes_data.rename(columns={CVs[0]: 'cv1', CVs[1]: 'cv2', 'file.free': 'val'},
                    inplace=True)
    # Convert the CVs to Angstroms:
    fes_data.cv1 = fes_data.cv1.multiply(10)
    fes_data.cv2 = fes_data.cv2.multiply(10)
    # Isolate the values that correspond to the basins
    basin_A = fes_data[(fes_data.cv1.between(A[0], A[1])) & (fes_data.cv2.between(A[2], A[3]))].val
    basin_B = fes_data[(fes_data.cv1.between(B[0], B[1])) & (fes_data.cv2.between(B[2], B[3]))].val
    # Calculate the dG from the minimum value in each basin (bound - unbound)
    delta_g = basin_A.min() - basin_B.min()
    # Convert to kcal and apply volume correction for funnel
    delta_g = (delta_g / 4.184) + vol_corr
    return delta_g
