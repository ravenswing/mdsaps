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


def cut_traj(trj_path, tpr, out_path, dt=100, ndx='i.ndx'):
    """ cutdown the trajectories using Gromacs trjconv ready for GISMO """
    # Assume working directory is same as traj if not specified
    tpr = tpr if '/' in tpr else '/'.join(trj_path.split('/')[:-1]) + '/' + tpr
    out_path = out_path if '/' in out_path else '/'.join(trj_path.split('/')[:-1]) + '/' + out_path
    ndx = ndx if '/' in ndx else '/'.join(trj_path.split('/')[:-1]) + '/' + ndx
    # Create the trjconv command from user input
    cmd = ("echo Backbone Protein_LIG | gmx_mpi trjconv "
           f"-s {tpr} "
           f"-f {trj_path} "
           f"-o {out_path} "
           f"-n {ndx} "
           "-fit rot+trans "
           f"-dt {dt} ")
    # Run the trjconv command
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
    assert len(bound) == 2, 'Bound requires 2 inputs'
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
    rx = [1.]+[t for t in data.loc[rx_ind[1:]].t.tolist()][1::2]
    # Count number of recrossings
    N = int((len(rx)-1))
    # Output number of RX and list of RX times
    return N, rx


def rx(dir_dict, var,
        bound=None, unbound=None, from_colvar=None, from_hdf=None,
        outpath='rx', columns=['system', 'lig', 'rep']):

    # Create storage df
    data = pd.DataFrame(columns=columns + ['number', 'rx'])

    for wd, ids in dir_dict.items():
        # Check that the ids and output columns match shape 
        assert len(ids) == len(columns), 'IDs given do not match columns.'
        # Check if both data sources are given
        if from_colvar is not None and from_hdf is not None:
            raise Exception('from_hdf and from_colvar are mutually exclusive')
        # Load a format colvar data
        elif from_colvar is not None:
            # Load the colvar into DataFrame
            df = load.colvar(f"{wd}/{from_colvar}", 'as_pandas')
            # Rename time column
            df.rename(columns={'time': 't'}, inplace=True)
            # Remove unnecessary columns
            df.drop(columns=[col for col in df if col not in ['t', var]],
                    inplace=True)
            # Convert CVs to Angstroms
            df[var] = df[var].multiply(10)
        # Load and format stored HDF data
        elif from_hdf is not None:
            # Load hdf file
            df = pd.read_hdf(from_hdf, key='df')
            # Collapse multi-indexing
            df.columns = df.columns.map('-'.join)
            # Column name to match, based on ids
            var = '-'.join(ids)
            # Create time column from df index
            df['t'] = df.index.to_series()
            # Remove unnecessary columns
            df = df.drop(columns=[col for col in df if col not in ['t', var]])
        # Check if any source provided
        else:
            raise Exception('no source for data')
        # Convert time to ns
        df.t = df.t.multiply(0.001)
        # Identify bound state from initial CV value if none given.
        if bound is None:
            bnd = [df[var].iloc[0], 1.0]
        # If only int is given assume +/- = 0
        elif isinstance(bound, float):
            bnd = [bound, 0.0]
        elif isinstance(bound, list):
            bnd = bound
        # Check if bound is either a list or an int
        else:
            raise Exception("Bound must be float or list")
        # Identify unbound state from max CV value - 10 if none given.
        unb = df[var].max() - 10 if unbound is None else unbound
        # Run recrossing counting function
        N, rx = _identify_recross(df, var, bound=bnd, unbound=unb)
        # Add values to storage dataframe
        data.loc[len(data.index)] = ids + [N, rx]
    # Save data
    data.to_hdf(f"{outpath}.h5", key='df', mode='w')


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


def reconstruct_traj(trj_path, tpr, out_path=None, ndx='i.ndx',
                     out_group='System'):
    # Assume working directory is same as traj if not specified
    tpr = tpr if '/' in tpr else '/'.join(trj_path.split('/')[:-1]) + '/' + tpr
    ndx = ndx if '/' in ndx else '/'.join(trj_path.split('/')[:-1]) + '/' + ndx
    if out_path is None:
        out_path = (f"{'/'.join(trj_path.split('/')[:-1])}/"
                    f"{trj_path.split('/')[-1][:-4]}_final.xtc")
    elif '/' not in out_path:
        out_path = '/'.join(trj_path.split('/')[:-1]) + '/' + out_path
    # Step 1: -pbc whole, produces tmp1.xtc
    try:
        subprocess.run((f"echo {out_group} | gmx_mpi trjconv "
                        f"-f {trj_path} "
                        f"-s {tpr} "
                        f"-n {ndx} "
                        "-o /tmp/tmp1.xtc "
                        '-pbc whole '),
                       check=True,
                       shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Step 2: -pbc cluster, produces tmp2.xtc
    try:
        subprocess.run((f"echo Protein {out_group} | gmx_mpi trjconv "
                        "-f /tmp/tmp1.xtc "
                        f"-s {tpr} "
                        f"-n {ndx} "
                        "-o /tmp/tmp2.xtc "
                        '-pbc cluster '),
                       check=True,
                       shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # run trjconv to produce a readable output
    try:
        subprocess.run((f"echo Protein {out_group} | gmx_mpi trjconv "
                        f"-f /tmp/tmp2.xtc "
                        f"-s {tpr} "
                        f"-n {ndx} "
                        f"-o {out_path} "
                        '-pbc mol -ur compact -center'),
                       check=True,
                       shell=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Remove temp xtc files if necessary
    subprocess.run("rm /tmp/*.xtc", shell=True)
