import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import MDAnalysis.transformations as trans
import pandas as pd
from os.path import exists
import argparse

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('top_path', type=str)
parser.add_argument('trj_path', type=str)
parser.add_argument('hdf', type=str)
parser.add_argument('measure', type=str)
parser.add_argument('select', type=str)
parser.add_argument('-res', action="store_true")
parser.add_argument('-ids', nargs='+', required=True)
args = parser.parse_args()
hdf_path = args.hdf
ids = args.ids


for i in ids:
    print(i)


def measure_rmsf(top_path, trj_path, measure="backbone", select="protein",
                 per_res=True):
    # Load new universe (N.B. TO MEMORY!)
    print('loading u')
    u = mda.Universe(top_path, trj_path, in_memory=True)
    print('finished loading u')

    # Check that trajectory is not too large (for RAM = 16 GB)

    assert len(u.trajectory) < 50002, "Trajectory too large"
    # Select the protein atoms, or some subset
    protein = u.select_atoms(select)

    # 1) the trajectory may be split across periodic boundaries,
    #    so we first make the protein whole and center it in the
    #    box using on-the-fly transformations
    print('making transformations')
    not_protein = u.select_atoms(f"not ({select})")
    transforms = [trans.center_in_box(protein, wrap=True),
                  trans.wrap(not_protein)]
    u.trajectory.add_transformations(*transforms)
    print('finished transformations')

    # 2) fit to the initial frame to get a better average structure
    #    (the trajectory is changed in memory)
    print('pre-aligning')
    prealigner = align.AlignTraj(u, u, select="protein and name CA",
                                 in_memory=True).run()
    print('finished pre-aligning')
    # 3) determine the average structure to use as a reference for
    #    the RMSF calculations, and align to the reference
    print('aligning')
    ref_coordinates = u.trajectory.timeseries(asel=protein).mean(axis=1)
    reference = mda.Merge(protein).load_new(ref_coordinates[:, None, :],
                                            order="afc")
    aligner = align.AlignTraj(u, reference,
                              select="protein and name CA",
                              in_memory=True).run()
    print('finished aligning')
    # 4) run the RMSF for the selected set of atoms
    backbone = protein.select_atoms(measure)
    rmsf = rms.RMSF(backbone, verbose=True).run()
    # 5) return rmsf values in a dataframe, averaging for residue
    #    number if the per_res flag is given.
    df = pd.DataFrame({'res': backbone.resnums, 'rmsf': rmsf.results.rmsf})

    return pd.DataFrame(df.groupby('res').rmsf.mean()) if per_res else df


new_data = measure_rmsf(args.top_path,
                        args.trj_path,
                        args.measure,
                        args.select,
                        args.res)

# Rename and stack for MultiIndexing
new_data.rename(columns={'rmsf': ids[2]}, inplace=True)
inp_l = pd.concat({ids[1]: new_data}, axis=1)
inp_s = pd.concat({ids[0]: inp_l}, axis=1)

# Add the data to the HDF stored data:
# If there is already an hdf file
if exists(hdf_path):
    print('Further time --> Reading Files & Adding Data')
    new = pd.read_hdf(hdf_path, key='df')
    # Update the values if the data already exists
    if any([(mi == inp_s.columns)[0] for mi in new.columns]):
        print("Updating values in DataFrame.")
        new.update(inp_s)
    # Or add the new data
    else:
        print("Adding new values to DataFrame.")
        new = new.join(inp_s)
    # Reorder the columns before saving the data
    new = new.iloc[:, new.columns.sortlevel(0, sort_remaining=True)[1]]
    # Write the new data to the existing file
    new.to_hdf(hdf_path, key='df')
# Or create one
else:
    # Make a new hdf file and save the first column of data
    print('First time --> Creating Files')
    inp_s.to_hdf(hdf_path, key='df')
