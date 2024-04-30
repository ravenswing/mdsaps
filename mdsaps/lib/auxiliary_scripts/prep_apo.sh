#!/bin/bash

# ----------  SYSTEM BUILING  ----------

export name=$(cd ..; basename -- "$PWD")
export GMX=gmx_mpi

# build dodecahedral box w/ 1.2 nm clearance
$GMX editconf -f ${name}_built.gro -o ${name}_box.gro -c -d 1.2 -bt dodecahedron

# solvate the system (default: TIP3P)
$GMX solvate -cp ${name}_box.gro -o ${name}_sol.gro -p $name.top

 # add ions to neutralise or to reach 0.15 moldm-3 NaCl concentration
$GMX grompp -f prep.mdp -c ${name}_sol.gro -p $name.top -o ions.tpr -maxwarn 1
#echo SOL | $GMX genion -s ions.tpr -o ${name}.gro -p $name.top -pname NA -nname CL -neutral
echo SOL | $GMX genion -s ions.tpr -o ${name}.gro -p $name.top -pname NA -nname CL -neutral -conc 0.15

# generate alpha-Carbon position restraints
echo C-alpha | $GMX genrestr -f ${name}.gro -o posres_CAlpha.itp

# generate index file
echo -e "q" | $GMX make_ndx -f ${name}.gro -o i.ndx

# reshape to visualise
$GMX grompp -f prep.mdp -c ${name}.gro -p $name.top -o reshape.tpr -n i.ndx
echo System | $GMX trjconv -f ${name}.gro -s reshape.tpr -o Readable_${name}.gro -pbc mol -ur compact -n i.ndx
