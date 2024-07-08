########################################################
#
# Reweight script based on the algorithm proposed by
# Tiwary and Parrinello JPCB 2014
#
# Check www.ucl.ac.uk/chemistry/research/group_pages/prot_dynamics
# for the most updated version.
#
# L. Sutto
# l.sutto@ucl.ac.uk
# f.l.gervasio@ucl.ac.uk               v1.0 - 23/04/2015
# ladislav.hovan.15@ucl.ac.uk          v2.0 - 30/01/2019
########################################################

import os.path
import argparse
import numpy as np
from math import exp, ceil


d = """
========================================================================
Time-independent Free Energy reconstruction script (a.k.a. reweight)
based on the algorithm proposed by Tiwary and Parrinello JPCB 2014

Typical usages:

1) to project your metadynamics FES on CVs you did not
   bias during your metadynamics run

2) to estimate the error on your FE profiles by comparing them with
   the FE profiles obtained integrating the metadynamics bias
   e.g. using plumed sum_hills


Example:

reweight.py -bsf 5.0 -kt 2.5 -fpref fes2d- -nf 80 -fcol 3 
            -colvar COLVAR -biascol 4 -rewcol 2 3

takes as input 80 FES files: fes2d-0.dat, fes2d-1.dat, ..., fes2d-79.dat 
obtained using a well-tempered metadynamics with bias factor 5
and containing the free energy in the 3rd column and the COLVAR file
containing the bias in the 4th column and outputs the FES projected 
on the CVs in column 2 and 3 of COLVAR file.

Ludovico Sutto
l.sutto@ucl.ac.uk                                       v1.0 - 23/04/2015

Added Python 3 compatibility, loading from and saving of ebetac files
which allows sidestepping the fes files for repeated reweighting from
the same simulation, and the option to specify bin size for each CV
independently.

New options: -ebetac <filename>, -savelist <filename>

Ladislav Hovan
ladislav.hovan.15@ucl.ac.uk                             v2.0 - 30/01/2019
=========================================================================
"""

# parser = argparse.ArgumentParser(
#    formatter_class=argparse.RawDescriptionHelpFormatter, description=d, epilog=" "
# )


def parse_args(*args):
    parser = argparse.ArgumentParser(description=d)

    add_input_args(parser)
    add_output_args(parser)
    add_data_args(parser)

    return parser.parse_args(*args)


def add_input_args(parser):
    group = parser.add_argument_group(
        "Input Options", "Options related to reading inputs."
    )
    group.add_argument(
        "-bsf",
        type=float,
        help="biasfactor used in the well-tempered metadynamics, if omitted assumes a non-well-tempered metadynamics",
    )
    group.add_argument(
        "-kt",
        type=float,
        default="2.49",
        help="kT in the energy units of the FES files (default: %(default)s)",
    )
    group.add_argument(
        "-fpref",
        default="fes",
        help="FES filenames prefix as generated with plumed sum_hills --stride. Expects FPREF%%d.dat (default: %(default)s)",
    )
    group.add_argument(
        "-nf",
        type=int,
        default=100,
        help="number of FES input files (default: %(default)s)",
    )
    group.add_argument(
        "-fcol",
        type=int,
        default=2,
        help="free energy column in the FES input files (first column = 1) (default: %(default)s)",
    )
    group.add_argument(
        "-ebetac", help="use precalculated ebetac list, if omitted use FES files"
    )
    group.add_argument(
        "-colvar",
        default="COLVAR",
        help="filename containing original CVs, reweighting CVs and metadynamics bias",
    )
    group.add_argument(
        "-rewcol",
        type=int,
        nargs="+",
        default=[2],
        help="column(s) in colvar file containing the CV to be reweighted (first column = 1) (default: %(default)s)",
    )
    group.add_argument(
        "-biascol",
        type=int,
        nargs="+",
        default=[4],
        help="column(s) in colvar file containing any energy bias (metadynamic bias, walls, external potentials..) (first column = 1) (default: %(default)s)",
    )


def add_output_args(parser):
    group = parser.add_argument_group(
        "Output Options", "Options related to saving files."
    )
    group.add_argument("-savelist", help="save ebetac list into this file")
    group.add_argument(
        "-outfile",
        default="fes_rew.dat",
        help="output FES filename (default: %(default)s)",
    )


def add_data_args(parser):
    group = parser.add_argument_group(
        "Extra Data Options", "Options related to calculations."
    )
    group.add_argument(
        "-min",
        type=float,
        nargs="+",
        help="minimum values of the CV in colvar file, if omitted find it",
    )
    group.add_argument(
        "-max",
        type=float,
        nargs="+",
        help="maximum values of the CV in colvar file, if omitted find it",
    )
    group.add_argument(
        "-bin",
        type=int,
        nargs="+",
        default=[50],
        help="number of bins for the reweighted FES (default: %(default)s for each CV)",
    )
    group.add_argument("-v", "--verbose", action="store_true", help="be verbose")


########################################################
# PARSING INPUTS
########################################################

args = parse_args()

# Well-Tempered Metadynamics or not
if args.bsf is not None and args.bsf > 0:
    tempered = True
else:
    tempered = False

# biasfactor for Well-Tempered
gamma = args.bsf

# kT in energy units (kJ or kcal)
kT = args.kt

# ebetac file for loading
ebetacFile = args.ebetac

# ebetac file for saving
ebetacSave = args.savelist

# input FES file prefix
fesfilename = args.fpref

# number of FES file generated with sum_hills stride option
# the more the better
numdat = args.nf

# column in FES file corresponding to the Free Energy
# NB: the first column is 0
col_fe = args.fcol - 1

# name of the file containing the CVs on which to project the FES and the bias
datafile = args.colvar

# list with the columns of the CVs on which to project the FES
# NB: the first column is 0
col_rewt = [i - 1 for i in args.rewcol]
numrewt = len(col_rewt)

# list with column numbers of your datafile containing the bias
# and any external bias/restraint/walls
# NB: the first column is 0
col_bias = [i - 1 for i in args.biascol]

# NB: if I don't define -min or -max in the input, I will find their value scanning the COLVAR file
s_min = args.min
s_max = args.max

# grid size for the reweighted FES
ngrid = args.bin
if len(ngrid) != numrewt:
    if len(ngrid) == 1:
        ngrid = [ngrid[0] for i in col_rewt]
    else:
        print(
            "ERROR: the number of grid dimensions (%d) does not match the number of reweighting CVs (%d)"
            % (len(ngrid), numrewt)
        )
        exit(1)

# output FES filename
out_fes_xy = args.outfile

# print some output while running
verbose = args.verbose


# CHECK IF NECESSARY FILES EXIST BEFORE STARTING
def verify_inputs():

    if not os.path.isfile(datafile):
        print("ERROR: file %s not found, check your inputs" % datafile)
        exit(1)
    if ebetacFile:
        if not os.path.isfile(ebetacFile):
            print("ERROR: file %s not found, check your inputs" % ebetacFile)
            exit(1)
    else:
        for i in range(numdat):
            fname = "%s%d.dat" % (fesfilename, i)
            if not os.path.isfile(fname):
                print("ERROR: file %s not found, check your inputs" % fname)
                exit(1)

verify_inputs()


########################################################
# FIRST PART: calculate c(t)
# This part is independent on the number of CVs being biased
# c(t) represents an estimate of the reversible
# work performed on the system until time t
########################################################

if verbose:
    print("Reading FES files..")

# calculates ebetac = exp(beta c(t)), using eq. 12 in eq. 3 in the JPCB paper
if ebetacFile:
    ebetac = list(np.loadtxt(ebetacFile))
else:
    ebetac = []
    for i in range(numdat):
        if verbose and numdat > 10 and i % (numdat // 10) == 0:
            print("%d of %d (%.0f%%) done" % (i, numdat, (i * 100.0 / numdat)))

        ########################################
        # set appropriate format for FES file names, NB: i starts from 0
        fname = "%s%d.dat" % (fesfilename, i)
        # fname = '%s.%d' % (fesfilename,i+1)
        ########################################

        data = np.loadtxt(fname)
        s1, s2 = 0.0, 0.0
        if tempered:
            for p in data:
                exponent = -p[col_fe] / kT
                s1 += exp(exponent)
                s2 += exp(exponent / gamma)
        else:
            for p in data:
                s1 += exp(-p[col_fe] / kT)
            s2 = len(data)
        ebetac += (s1 / s2,)

# this would be c(t):
# coft = [ kT*log(x) for x in ebetac ]

if ebetacSave:
    np.savetxt("backup_ebetac.dat", ebetacSave)

########################################################
# SECOND PART: Boltzmann-like sampling for reweighting
########################################################

if verbose:
    print("Calculating CV ranges..")

# NB: loadtxt takes care of ignoring comment lines starting with '#'
colvar = np.loadtxt(datafile)

# find min and max of rew CV
numcolv = 0
calc_smin = False
calc_smax = False

if not s_min:
    s_min = [9e99] * numrewt
    calc_smin = True
if not s_max:
    s_max = [-9e99] * numrewt
    calc_smax = True

for row in colvar:
    numcolv += 1

    for i in range(numrewt):
        col = col_rewt[i]
        val = row[col]

        if calc_smin:
            if val < s_min[i]:
                s_min[i] = val
        if calc_smax:
            if val > s_max[i]:
                s_max[i] = val

if verbose:
    for i in range(numrewt):
        print("CV[%d] range: %10.5f ; %10.5f" % (i, s_min[i], s_max[i]))

# build the new square grid for the reweighted FES
s_grid = [[]] * numrewt
for i in range(numrewt):
    ds = (s_max[i] - s_min[i]) / (ngrid[i] - 1)
    s_grid[i] = [s_min[i] + n * ds for n in range(ngrid[i])]
    if verbose:
        print("Grid ds CV[%d]=%f" % (i, ds))

if verbose:
    print("Calculating reweighted FES..")

# initialize square array numrewt-dimensional
fes = np.zeros(ngrid)

# go through the CV(t) trajectory
denom = 0.0
i = 0
for row in colvar:
    i += 1

    # build the array of grid indeces locs corresponding to the point closest to current point
    locs = [[]] * numrewt
    for j in range(numrewt):
        col = col_rewt[j]
        val = row[col]
        diff = np.array([abs(gval - val) for gval in s_grid[j]])
        locs[j] = [diff.argmin()]  # find position of minimum in diff array

    # find closest c(t) for this point of time
    indx = int(ceil(float(i) / numcolv * numdat)) - 1

    bias = sum([row[j] for j in col_bias])
    ebias = exp(bias / kT) / ebetac[indx]
    fes[locs] += ebias
    denom += ebias

# ignore warnings about log(0) and /0
np.seterr(all="ignore")

fes /= denom
fes = -kT * np.log(fes)

# set FES minimum to 0
fes -= np.min(fes)


########################################################
# OUTPUT RESULTS ON FILE
########################################################

if verbose:
    print("Saving results on %s" % out_fes_xy)

# save the FES in the format: FES(x,y) (one increment of y per row)
# np.savetxt('fes_rew_matlabfmt.dat', fes, fmt='%.8e', delimiter=' ')

# print the FES in the format:
# x,y,z,FES(x,y,z) for 3D
# x,y,FES(x,y) for 2D
# x,FES(x) for 1D
with open(out_fes_xy, "w") as f:
    if numrewt == 3:
        for nz, z in enumerate(s_grid[2]):
            for ny, y in enumerate(s_grid[1]):
                for nx, x in enumerate(s_grid[0]):
                    f.write(
                        "%20.12f %20.12f %20.12f %20.12f\n" % (x, y, z, fes[nx][ny][nz])
                    )
                f.write("\n")
    elif numrewt == 2:
        for ny, y in enumerate(s_grid[1]):
            for nx, x in enumerate(s_grid[0]):
                f.write("%20.12f %20.12f %20.12f\n" % (x, y, fes[nx][ny]))
            f.write("\n")
    elif numrewt == 1:
        for nx, x in enumerate(s_grid[0]):
            f.write("%20.12f %20.12f\n" % (x, fes[nx]))
f.close()

def main() -> None:



if __name__ == '__main__':
    main()
