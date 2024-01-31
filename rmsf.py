import sys
sys.path.append('/home/rhys/phd_tools/SAPS/')
from traj_tools import _init_universe

import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import MDAnalysis.transformations as trans

DATA_DIR = '/home/rhys/Storage/ampk_metad_all_data'
method='fun-metaD'
system='a2b1'
pdb='A769'
rep='R2'
wd = f"{DATA_DIR}/{method}/{system}+{pdb}/{rep}"

#f"{wd}/{system}+{pdb}_{rep}_GISMO.xtc"

u = mda.Universe(f"{wd}/md_dry.pdb", f"{wd}/{system}+{pdb}_short.xtc", in_memory=True)

#u.trajectory[::5]
#print(len(u.trajectory))
protein = u.select_atoms("protein or resid 303")

# 1) the current trajectory contains a protein split across
#    periodic boundaries, so we first make the protein whole and
#    center it in the box using on-the-fly transformations
not_protein = u.select_atoms('not (protein or resid 303)')
transforms = [trans.center_in_box(protein, wrap=True),
              trans.wrap(not_protein)]
u.trajectory.add_transformations(*transforms)


# 2) fit to the initial frame to get a better average structure
#    (the trajectory is changed in memory)
prealigner = align.AlignTraj(u, u, select="protein and name CA",
                             in_memory=True).run()

# 3) reference = average structure
ref_coordinates = u.trajectory.timeseries(asel=protein).mean(axis=1)
# make a reference structure (need to reshape into a 1-frame
# "trajectory")
reference = mda.Merge(protein).load_new(ref_coordinates[:, None, :],
                                        order="afc")

aligner = align.AlignTraj(u, reference,
                          select="protein and name CA",
                          in_memory=True).run()

backbone = protein.select_atoms("backbone")
rmsfer = rms.RMSF(backbone, verbose=True).run()
