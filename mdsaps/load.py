"""
===============================================================================
                                LOADING DATA
===============================================================================
"""

import numpy as np
import pandas as pd
import pickle


def hills(filename: str) -> list:
    hills = [[], []]
    with open(filename) as f:
        lines = f.readlines()
    data = [s for s in lines if not s.startswith(('@', '#'))]
    data = [[float(val) for val in ln.split()] for ln in data]
    hills[0] = ([ln[0] for ln in data])
    hills[1] = ([ln[5] for ln in data])
    return hills


def colvar(filename: str, output: str='as_pandas'):
    with open(filename) as f:
        head = f.readlines()[0]
    head = head.split()[2:]
    # Read in old COLVAR file into DataFrame.
    # Filters out comment lines and splits columns via whitespace.
    old_col = pd.concat([df for df in
                         pd.read_csv(filename,
                                     delim_whitespace=True,
                                     names=head,
                                     skiprows=1,
                                     comment='#',
                                     chunksize=1000)])
    # Round the timestamps to ensure successful merging
    old_col['int_time'] = old_col['time'].astype(float).astype(int)
    # Remove duplicate lines created by restarts
    old_col = old_col.drop_duplicates(subset='int_time', keep='last')
    if output == 'as_pandas':
        return old_col
    elif output == 'as_numpy':
        return old_col.values.astype(float)


def fes(filename: str, mode: str = 'pandas', is_rew: bool = False):
    if mode == 'pandas':
        with open(filename) as f:
            head = f.readlines()[0]
        head = head.split()[2:]
        fes = pd.concat([df for df in pd.read_csv(filename,
                                                  delim_whitespace=True,
                                                  names=head,
                                                  comment='#',
                                                  chunksize=1000)])
        return fes
    else:
        fes = [[], [], []]
        with open(filename) as f:
            lines = f.readlines()
            data = [s for s in lines if not s.startswith(('@', '#'))]
        data = [[float(val) for val in ln.split()] for ln in data]
        breaks = [i for i, e in enumerate(data) if e == []]
        # Remove blank lines
        for index in sorted(breaks, reverse=True):
            del data[index]
        # Get the number of bins and CV names from header
        if is_rew:
            nbins = int(breaks[0])
            [x_name, y_name] = ['RMSD to IN', 'RMSD to OUT']
        else:
            nbins = int([line.split()[-1] for line in lines
                        if 'nbins_pp.proj' in line][0])
            [x_name, y_name] = lines[0].split()[2:4]
        # Organise data into plotable arrays
        split_data = [data[i:i + nbins] for i in range(0, len(data), nbins)]
        z = []
        for block in split_data:
            z.append([line[2] for line in block])
            fes[1].append(block[0][1])
        fes[0] = [line[0] for line in split_data[0]]
        fes[2] = np.asarray(z)
        return fes, [x_name, y_name]


def xvg(filename: str) -> list:
    ''' load xvg '''
    with open(filename) as f:
        lines = f.readlines()
        data = [s for s in lines if not s.startswith(('@', '#'))]
    data = [[float(val) for val in ln.split()] for ln in data]
    return data


def cd(filename: str) -> list:
    ''' load CD data '''
    with open(filename) as f:
        lines = f.readlines()
        data = [s for s in lines if not s.startswith(('@', '#'))]
    data = [[float(val) for val in ln.split()] for ln in data]
    # data = [line for line in data if len(line) == 2]
    # if "SESCA" in filename: print(data)
    # out_data = [[l[0] for l in data], [l[1] for l in data]]
    # return out_data
    return data


def p(filename: str):
    ''' Simple 1D pickle load '''
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pdb(filename: str) -> list:
    # Load complex pdb to edit
    with open(filename, 'r') as f:
        lines = f.readlines()
    print(f'Loaded {filename}')
    lines = [line.split() for line in lines]
    return lines
