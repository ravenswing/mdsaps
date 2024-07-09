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
        help="number of bins for the reweighted FES (default: %(default)s for each CV)",
    )
    group.add_argument("-v", "--verbose", action="store_true", help="be verbose")


def setup_global_variables() -> None:
    # kT in energy units (kJ or kcal)
    global kT
    kT = args.kt

    # biasfactor for Well-Tempered
    global gamma
    gamma = args.bsf

    # Well-Tempered Metadynamics or not
    global is_well_tempered
    if args.bsf is not None and args.bsf > 0:
        is_well_tempered = True
    else:
        is_well_tempered = False

    # print some output while running
    global verbose
    verbose = args.verbose


# CHECK IF NECESSARY FILES EXIST BEFORE STARTING
def verify_inputs(colvar_file, exp_beta_ct_file, num_fes_files, fes_file_prefix):
    if not os.path.isfile(colvar_file):
        print("ERROR: file %s not found, check your inputs" % colvar_file)
        exit(1)
    if exp_beta_ct_file:
        if not os.path.isfile(exp_beta_ct_file):
            print("ERROR: file %s not found, check your inputs" % exp_beta_ct_file)
            exit(1)
    else:
        for i in range(num_fes_files):
            fname = "%s%d.dat" % (fes_file_prefix, i)
            if not os.path.isfile(fname):
                print("ERROR: file %s not found, check your inputs" % fname)
                exit(1)


# FIRST PART: calculate c(t)
def calculate_ct(num_fes_files, fes_file_prefix, fes_column_free) -> None:
    # This part is independent on the number of CVs being biased
    # c(t) represents an estimate of the reversible
    # work performed on the system until time t
    if verbose:
        print("Reading FES files...")

    # calculates ebetac = exp(beta c(t)), using eq. 12 in eq. 3 in the JPCB paper
    ebetac = []
    for i in range(num_fes_files):
        if verbose and num_fes_files > 10 and i % (num_fes_files // 10) == 0:
            print(
                "%d of %d (%.0f%%) done"
                % (i, num_fes_files, (i * 100.0 / num_fes_files))
            )

        ########################################
        # set appropriate format for FES file names, NB: i starts from 0
        fname = "%s%d.dat" % (fes_file_prefix, i)
        # fname = '%s.%d' % (fes_file_prefix,i+1)
        ########################################

        data = np.loadtxt(fname)
        s1, s2 = 0.0, 0.0
        if is_well_tempered:
            for p in data:
                exponent = -p[fes_column_free] / kT
                s1 += exp(exponent)
                s2 += exp(exponent / gamma)
        else:
            for p in data:
                s1 += exp(-p[fes_column_free] / kT)
            s2 = len(data)
        ebetac.append(s1 / s2)

    # this would be c(t):
    # coft = [ kT*log(x) for x in ebetac ]

    return ebetac


def calculate_cv_ranges(colvar_file, rew_dimension, colvar_rew_columns, s_min, s_max):
    if verbose:
        print("Calculating CV ranges..")

    # NB: loadtxt takes care of ignoring comment lines starting with '#'
    colvar = np.loadtxt(colvar_file)

    # find min and max of rew CV
    calc_smin = False
    calc_smax = False

    if not s_min:
        s_min = [9e99] * rew_dimension
        calc_smin = True
    if not s_max:
        s_max = [-9e99] * rew_dimension
        calc_smax = True

    for row in colvar:
        for i in range(rew_dimension):
            col = colvar_rew_columns[i]
            val = row[col]

            if calc_smin:
                if val < s_min[i]:
                    s_min[i] = val
            if calc_smax:
                if val > s_max[i]:
                    s_max[i] = val

    if verbose:
        for i in range(rew_dimension):
            print("CV[%d] range: %10.5f ; %10.5f" % (i, s_min[i], s_max[i]))

    return s_min, s_max


# SECOND PART: Boltzmann-like sampling for reweighting
def boltzmann_sampling(
    colvar_file,
    s_min,
    s_max,
    rew_dimension,
    grid_shape,
    colvar_rew_columns,
    num_fes_files,
    colvar_bias_columns,
    ebetac,
) -> None:
    # Load the colvar file into a numpy array
    # NB: loadtxt takes care of ignoring comment lines starting with '#'
    colvar = np.loadtxt(colvar_file)

    # Build the new square grid for the reweighted FES
    s_grid = [[]] * rew_dimension
    for i in range(rew_dimension):
        ds = (s_max[i] - s_min[i]) / (grid_shape[i] - 1)
        s_grid[i] = [s_min[i] + n * ds for n in range(grid_shape[i])]
        if verbose:
            print("Grid ds CV[%d]=%f" % (i, ds))

    if verbose:
        print("Calculating reweighted FES..")

    # initialize square array rew_dimension-dimensional
    fes = np.zeros(grid_shape)

    # go through the CV(t) trajectory
    denom = 0.0

    for i, row in enumerate(colvar):
        # build the array of grid indeces locs corresponding to the point closest to current point
        locs = [] * rew_dimension

        for j in range(rew_dimension):
            col = colvar_rew_columns[j]
            val = row[col]
            diff = np.array([abs(gval - val) for gval in s_grid[j]])
            locs.append(diff.argmin())  # find position of minimum in diff array

        locs = tuple(locs)

        # find closest c(t) for this point of time
        indx = int(ceil(float(i) / len(colvar) * num_fes_files)) - 1

        bias = sum([row[j] for j in colvar_bias_columns])
        ebias = exp(bias / kT) / ebetac[indx]
        fes[locs] += ebias
        denom += ebias

    # ignore warnings about log(0) and /0
    np.seterr(all="ignore")

    fes /= denom

    fes = -kT * np.log(fes)

    # set FES minimum to 0
    fes -= np.min(fes)

    return fes, s_grid


# OUTPUT RESULTS TO FILE
def save_output(output_file, rew_dimension, s_grid, fes) -> None:
    if verbose:
        print("Saving results on %s" % output_file)

    # save the FES in the format: FES(x,y) (one increment of y per row)
    # np.savetxt('fes_rew_matlabfmt.dat', fes, fmt='%.8e', delimiter=' ')

    # print the FES in the format:
    # x,y,z,FES(x,y,z) for 3D
    # x,y,FES(x,y) for 2D
    # x,FES(x) for 1D
    with open(output_file, "w") as f:
        if rew_dimension == 3:
            for nz, z in enumerate(s_grid[2]):
                for ny, y in enumerate(s_grid[1]):
                    for nx, x in enumerate(s_grid[0]):
                        f.write(
                            "%20.12f %20.12f %20.12f %20.12f\n"
                            % (x, y, z, fes[nx][ny][nz])
                        )
                    f.write("\n")
        elif rew_dimension == 2:
            for ny, y in enumerate(s_grid[1]):
                for nx, x in enumerate(s_grid[0]):
                    f.write("%20.12f %20.12f %20.12f\n" % (x, y, fes[nx][ny]))
                f.write("\n")
        elif rew_dimension == 1:
            for nx, x in enumerate(s_grid[0]):
                f.write("%20.12f %20.12f\n" % (x, fes[nx]))
    f.close()


def main() -> None:
    global args
    args = parse_args()

    setup_global_variables()

    ### INPUT ARGUMENTS
    # Prefix for the input FES files (before number.dat)
    fes_file_prefix = args.fpref
    # Number of FES files generated with sum_hills stride option (the more the better)
    num_fes_files = args.nf
    # Column in FES file corresponding to the Free Energy
    # NB: the first column is 0
    fes_column_free = args.fcol - 1

    # Name of the file containing the CVs on which to project the FES and the bias
    colvar_file = args.colvar
    # List with the columns of the CVs on which to project the FES
    # NB: the first column is 0
    colvar_rew_columns = [i - 1 for i in args.rewcol]
    rew_dimension = len(colvar_rew_columns)
    # List with column numbers of your colvar_file containing the bias
    # and any external bias/restraint/walls --> CHECK
    # NB: the first column is 0
    colvar_bias_columns = [i - 1 for i in args.biascol]
    # Minimum and maximum bounds of the CVs in the input
    # NB: if I don't define -min or -max in the input, I will find their value scanning the COLVAR file
    s_min = args.min
    s_max = args.max

    # Optional: provide ebetac file for loading
    exp_beta_ct_file = args.ebetac

    ### OUTPUT ARGUMENTS
    # Output FES filename
    output_file = args.outfile
    # Optional: ebetac file for saving
    exp_beta_ct_save = args.savelist

    # Grid size for the reweighted FES
    if args.bin:
        assert (
            len(args.bin) == rew_dimension
        ), f"ERROR: the number of -bin provided ({len(args.bin)}) does not match the dimension of reweighting CVs ({rew_dimension})"
        grid_shape = args.bin
    else:
        grid_shape = [100] * rew_dimension

    verify_inputs(colvar_file, exp_beta_ct_file, num_fes_files, fes_file_prefix)

    if exp_beta_ct_file:
        exp_beta_ct = list(np.loadtxt(exp_beta_ct_file))
    else:
        exp_beta_ct = calculate_ct(
            num_fes_files,
            fes_file_prefix,
            fes_column_free,
        )

    if exp_beta_ct_save:
        np.savetxt(exp_beta_ct_save, exp_beta_ct)

    if all([s_min, s_max]):
        print("Minima and Maxima provided")
        cv_ranges_min = s_min
        cv_ranges_max = s_max
    else:
        cv_ranges_min, cv_ranges_max = calculate_cv_ranges(
            colvar_file, rew_dimension, colvar_rew_columns, s_min, s_max
        )

    fes, s_grid = boltzmann_sampling(
        colvar_file,
        cv_ranges_min,
        cv_ranges_max,
        rew_dimension,
        grid_shape,
        colvar_rew_columns,
        num_fes_files,
        colvar_bias_columns,
        exp_beta_ct,
    )

    save_output(output_file, rew_dimension, s_grid, fes)


if __name__ == "__main__":
    main()
