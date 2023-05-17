"""
===============================================================================
                        AMPK SPECIFIC Fun-METAD ANALYSIS
===============================================================================

    - Analysis
    - Plotting
    - Calculations
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import subprocess
import sys
from glob import glob
from math import ceil
sys.path.append('/home/rhys/phd_tools/python_scripts')
import graphics
import load_data as load
import plotly.graph_objects as go

import shutil
from math import floor

import MDAnalysis as mda
from MDAnalysis.analysis import diffusionmap, align, rms

sys.path.append('/home/rhys/phd_tools/SAPS')
import traj_tools as tt

ANG = "\u212B"

DATA_DIR = '/media/rhys/Storage/ampk_metad_all_data'
SYSTS = ['a2b1', 'a2b2']
LIGS = ['A769', 'PF739', 'SC4', 'MT47', 'MK87']

# Where to put the plots and other results
SAVE_DIR = '/home/rhys/Dropbox/RESEARCH/AA_RHYS/BB_AMPK/Fun-metaD_Results/Plots_wReplicas'


def fes_multiplot(cmax=32, replicas=False):
    if replicas:
        for rep in ['R'+str(x) for x in np.arange(3)+1]:
            fig, ax = plt.subplots(5, 2, figsize=(25, 30))
        #     fig.tight_layout(h_pad=4)
            t = 750 if rep == 'R1' else 500
            plt.suptitle(f'FES for {t}ns Fun-MetaD ({rep})')
            plt.subplots_adjust(top=0.94, right=0.915)
            funnel_parms = {'lw': 0.0,
                            'uw': 4.5,
                            'sc': 2.5,
                            'b': 1.0,
                            'f': 0.15,
                            'h': 1.5}
            i = 0
            for lig in LIGS:
                j = 0
                for system in SYSTS:
                    data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/{system}+{lig}_FES', False)
                    cmap = graphics.two_cv_contour(data, labels, cmax, ax[i, j], funnel_parms)
                    ax[i, j].set_title(f"{system}+{lig}")
                    ax[i, j].set_xlabel(f"{labels[0]} / nm")
                    ax[i, j].set_ylabel(f"{labels[1]} / nm")
                    j += 1
                i += 1
            cax = plt.axes([0.93, 0.11, 0.01, 0.77])
            cbar = plt.colorbar(cmap, cax=cax, aspect=10, ticks=np.arange(0., cmax+1, 2.0))
            cbar.set_label('Free Energy / kcal/mol', fontsize=10)
            fig.savefig(f'{SAVE_DIR}/FES_multi_{rep}.png', dpi=300, bbox_inches='tight')

    else:
        fig, ax = plt.subplots(4, 2, figsize=(25, 30))
    #     fig.tight_layout(h_pad=4)
        plt.suptitle('FES for 500ns Fun-MetaD')
        plt.subplots_adjust(top=0.94, right=0.915)
        funnel_parms = {'lw': 0.0,
                        'uw': 4.5,
                        'sc': 2.5,
                        'b': 1.0,
                        'f': 0.15,
                        'h': 1.5}
        i = 0
        for lig in LIGS:
            j = 0
            for system in SYSTS:
                data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{system}+{lig}_FES', False)
                cmap = graphics.two_cv_contour(data, labels, cmax, ax[i, j], funnel_parms)
                ax[i, j].set_title(f"{system}+{lig}")
                ax[i, j].set_xlabel(f"{labels[0]} / nm")
                ax[i, j].set_ylabel(f"{labels[1]} / nm")
                j += 1
            i += 1
        cax = plt.axes([0.93, 0.11, 0.01, 0.77])
        cbar = plt.colorbar(cmap, cax=cax, aspect=10, ticks=np.arange(0., cmax+1, 2.0))
        cbar.set_label('Free Energy / kcal/mol', fontsize=10)
        fig.savefig(f'{SAVE_DIR}/FES_multi.png', dpi=300, bbox_inches='tight')


def fes_strideplot(wd, name, cmax=32, stride=50, to_use=[0, 1, 2]):
    fig, ax = plt.subplots(1, len(to_use), figsize=(len(to_use*8), 6))
    # fig.tight_layout(h_pad=4)
    plt.suptitle(f"FES Changes over time for {name}")
    plt.subplots_adjust(top=0.85, right=0.915)
    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}
    max_vals = []
    for i in np.arange(len(to_use)):
        data, labels = load.fes(f'{wd}/{name}_FES{to_use[i]}.dat', False)
        data[2] = data[2]/4.184
        max_non_inf = np.amax(data[2][np.isfinite(data[2])])
        max_vals.append(max_non_inf)
        print('VMAX: ', max_non_inf)
    print(f"using: {max(max_vals)}")
    cmax = int(ceil(max_non_inf / 2.0)) * 2

    i = 0
    for i in np.arange(len(to_use)):
        data, labels = load.fes(f'{wd}/{name}_FES{to_use[i]}.dat', False)
        cmap = graphics.two_cv_contour(data, labels, cmax, ax[i], funnel_parms)
        ax[i].set_title(f"After {stride*(i+1)} ns")
        ax[i].set_xlabel(f"{labels[0]} / nm")
        ax[i].set_ylabel(f"{labels[1]} / nm")
        i += 1
    cax = plt.axes([0.93, 0.11, 0.01, 0.77])
    cbar = plt.colorbar(cmap, cax=cax, aspect=10,
                        ticks=np.arange(0., cmax, 2.0))
    cbar.set_label('Free Energy / kcal/mol', fontsize=10)
    fig.savefig(f'{SAVE_DIR}/FES_wStride_{name}.png', dpi=300,
                bbox_inches='tight')


def new_strideplot(wd, name, stride=50, to_use=[0, 1, 2], basins=None):
    fig, ax = plt.subplots(1, len(to_use), figsize=(len(to_use*8), 6))
    # fig.tight_layout(h_pad=4)
    plt.suptitle(f"FES Changes over time for {name}")
    plt.subplots_adjust(top=0.85, right=0.915)
    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}
    max_vals = []
    for i in np.arange(len(to_use)):
        data, labels = load.fes(f'{wd}/{name}_FES{to_use[i]}.dat', False)
        data[2] = data[2]/4.184
        max_non_inf = np.amax(data[2][np.isfinite(data[2])])
        max_vals.append(max_non_inf)
        print('VMAX: ', max_non_inf)
    print(f"using: {max(max_vals)}")
    cmax = max(max_vals)+1

    i = 0
    for i in np.arange(len(to_use)):
        data, labels = load.fes(f'{wd}/{name}_FES{to_use[i]}.dat', False)
        data[2] = data[2]/4.184
        max_non_inf = np.amax(data[2][np.isfinite(data[2])])
        data[2] = data[2] + (max(max_vals) - max_non_inf)
        data[2] = data[2]*4.184
        cmap = graphics.two_cv_contour(data, labels, cmax, ax[i], funnel_parms)
        ax[i].set_title(f"After {stride*(i+1)} ns")
        ax[i].set_xlabel(f"{labels[0]} / nm")
        ax[i].set_ylabel(f"{labels[1]} / nm")
        if basins is not None:
            b1 = plt.Rectangle((basins['bound'][0], basins['bound'][2]),
                               (basins['bound'][1] - basins['bound'][0]),
                               basins['bound'][3],
                               ls='--', fc='none', ec='k', lw=2.0)
            ax[i].add_patch(b1)
            b2 = plt.Rectangle((basins['unbound'][0], basins['unbound'][2]),
                               (basins['unbound'][1] - basins['unbound'][0]),
                               basins['unbound'][3],
                               ls='--', fc='none', ec='k', lw=2.0)
            ax[i].add_patch(b2)
        i += 1
    cax = plt.axes([0.93, 0.11, 0.01, 0.77])
    cbar = plt.colorbar(cmap, cax=cax, aspect=10,
                        ticks=np.arange(0., cmax, 2.0))
    cbar.set_label('Free Energy / kcal/mol', fontsize=10)
    fig.savefig(f'{SAVE_DIR}/NEW_FES_wStride_{name}.png', dpi=300,
                bbox_inches='tight')


def fes_by_replica(basins=None):
    for rep in ['R'+str(x) for x in np.arange(3)+1]:
        fig, ax = plt.subplots(5, 2, figsize=(25, 30))
        fig.tight_layout(h_pad=4)
        #t = 750 if rep == 'R1' else 500
        plt.suptitle(f'FES for Fun-MetaD ({rep})')
        plt.subplots_adjust(top=0.95, right=0.915)
        funnel_parms = {'lw': 0.0,
                        'uw': 4.5,
                        'sc': 2.5,
                        'b': 1.0,
                        'f': 0.15,
                        'h': 1.5}
        max_vals = []
        for lig in LIGS:
            for system in SYSTS:
                data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/{system}+{lig}_FES', False)
                data[2] = data[2]/4.184
                max_non_inf = np.amax(data[2][np.isfinite(data[2])])
                max_vals.append(max_non_inf)
                print('VMAX: ', max_non_inf)
        print(f"using: {max(max_vals)}")
        cmax = max(max_vals)+1

        i = 0
        for lig in LIGS:
            j = 0
            for system in SYSTS:
                data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/{system}+{lig}_FES', False)
                data[2] = data[2]/4.184
                max_non_inf = np.amax(data[2][np.isfinite(data[2])])
                data[2] = data[2] + (max(max_vals) - max_non_inf)
                data[2] = data[2]*4.184
                cmap = graphics.two_cv_contour(data, labels, cmax, ax[i, j], funnel_parms)
                with open(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/COLVAR', 'r') as f:
                    t = float(f.readlines()[-1].split()[0])/1000
                ax[i, j].set_title(f"{system}+{lig} (t = {t:.0f}ns)")
                ax[i, j].set_xlabel(f"{labels[0]} / nm")
                ax[i, j].set_ylabel(f"{labels[1]} / nm")
                if basins is not None:
                    b1 = plt.Rectangle((basins['bound'][0], basins['bound'][2]),
                                    (basins['bound'][1] - basins['bound'][0]),
                                    basins['bound'][3],
                                    ls='--', fc='none', ec='k', lw=2.0)
                    ax[i, j].add_patch(b1)
                    b2 = plt.Rectangle((basins['unbound'][0], basins['unbound'][2]),
                                    (basins['unbound'][1] - basins['unbound'][0]),
                                    basins['unbound'][3],
                                    ls='--', fc='none', ec='k', lw=2.0)
                    ax[i, j].add_patch(b2)
                j += 1
            i += 1
        cax = plt.axes([0.98, 0.11, 0.01, 0.77])
        cbar = plt.colorbar(cmap, cax=cax, aspect=10,
                            ticks=np.arange(0., cmax, 2.0))
        cbar.set_label('Free Energy / kcal/mol', fontsize=10)
        fig.savefig(f'{SAVE_DIR}/{rep}_FES_multi.png', dpi=300,
                    bbox_inches='tight')


def fes_by_system(ligand_list, system_list, n_reps, basins=None):
    for lig in ligand_list:
        for system in system_list:
            fig, ax = plt.subplots(1, 3, figsize=(24, 6))
            plt.suptitle(f'FES for Fun-MetaD ({system} + {lig})')
            plt.subplots_adjust(top=0.85, right=0.915)
            funnel_parms = {'lw': 0.0,
                            'uw': 4.5,
                            'sc': 2.5,
                            'b': 1.0,
                            'f': 0.15,
                            'h': 1.5}

            max_vals = []
            for rep in ['R'+str(x) for x in np.arange(n_reps)+1]:
                data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/{system}+{lig}_FES', False)
                data[2] = data[2]/4.184
                max_non_inf = np.amax(data[2][np.isfinite(data[2])])
                max_vals.append(max_non_inf)
                print('VMAX: ', max_non_inf)
            print(f"using: {max(max_vals)}")
            cmax = max(max_vals)+1

            i = 0
            for rep in ['R'+str(x) for x in np.arange(n_reps)+1]:
                data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/{system}+{lig}_FES', False)
                data[2] = data[2]/4.184
                max_non_inf = np.amax(data[2][np.isfinite(data[2])])
                data[2] = data[2] + (max(max_vals) - max_non_inf)
                data[2] = data[2]*4.184
                cmap = graphics.two_cv_contour(data, labels, cmax, ax[i], funnel_parms)
                with open(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/COLVAR', 'r') as f:
                    t = float(f.readlines()[-1].split()[0])/1000
                ax[i].set_title(f"{system}+{lig} (t = {t:.0f}ns)")
                ax[i].set_xlabel(f"{labels[0]} / nm")
                ax[i].set_ylabel(f"{labels[1]} / nm")
                if basins is not None:
                    b1 = plt.Rectangle((basins['bound'][0], basins['bound'][2]),
                                    (basins['bound'][1] - basins['bound'][0]),
                                    basins['bound'][3],
                                    ls='--', fc='none', ec='k', lw=2.0)
                    ax[i].add_patch(b1)
                    b2 = plt.Rectangle((basins['unbound'][0], basins['unbound'][2]),
                                    (basins['unbound'][1] - basins['unbound'][0]),
                                    basins['unbound'][3],
                                    ls='--', fc='none', ec='k', lw=2.0)
                    ax[i].add_patch(b2)
                i += 1

            cax = plt.axes([0.98, 0.11, 0.01, 0.77])
            cbar = plt.colorbar(cmap, cax=cax, aspect=10,
                                ticks=np.arange(0., cmax, 2.0))
            cbar.set_label('Free Energy / kcal/mol', fontsize=10)
            fig.savefig(f'{SAVE_DIR}/FES/{system}+{lig}_FES.png', dpi=450,
                        bbox_inches='tight')


def fes_highlight2(system, lig, rep, CLR, basins=None,):

    funnel_parms = {'lw': 0.0,
                    'uw': 4.5,
                    'sc': 2.5,
                    'b': 1.0,
                    'f': 0.15,
                    'h': 1.5}

    data, labels = load.fes(f'{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}/{system}+{lig}_FES', False)
    data[2] = data[2]/4.184
    cmax = np.amax(data[2][np.isfinite(data[2])])+1

    fig = go.Figure(data =
                    go.Contour(
                        z=data[2],
                        x=data[0], # horizontal axis
                        y=data[1], # vertical axis
                        colorscale='RdYlBu',
                        contours=dict(
                           start=0,
                           end=cmax,
                           size=2),
                        colorbar=dict(
                                    title='Free Energy (kcal/mol)',
                                    titleside='right')
                ))
    # format axes
    fig.update_xaxes(showline=True,
                        linecolor=CLR['ax'],
                        title_text = f"Funnel CV - Projection / nm",
                        linewidth=0.5,
                        title_standoff = 20,
                        ticks='outside', minor_ticks='outside')
    fig.update_yaxes(showline=True,
                        linecolor=CLR['ax'],
                        title_text = f"Funnel CV - Extension / nm",
                        linewidth=0.5,
                        title_standoff = 20,
                        ticks='outside', minor_ticks='outside')

    # format the rest of the figure
    fig.update_layout(height=1600, width=2200,
                      title_text="",
                      font=dict(color=CLR['ax'],
                                family='Arial', size=32),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)

    fig.add_shape(type="rect",
                  x0=basins[f"{lig}-bnd"][0]/10,
                  x1=basins[f"{lig}-bnd"][1]/10,
                  y0=basins[f"{lig}-bnd"][2]/10,
                  y1=basins[f"{lig}-bnd"][3]/10,
                  line_dash='dash',
                  line=dict(color=CLR['ax'], width=5))

    fig.add_shape(type="rect",
                  x0=basins[f"unbound"][0]/10,
                  x1=basins[f"unbound"][1]/10,
                  y0=basins[f"unbound"][2]/10,
                  y1=basins[f"unbound"][3]/10,
                  line_dash='dash',
                  line=dict(color=CLR['ax'], width=5))

    SAVE_DIR = '../Fun-metaD_Results/Plots_wReplicas'
    fig.write_image(f"{SAVE_DIR}/FES/New_Reps_Solo/{system}_{lig}_{rep}_soloFES.png", scale=2)

    # ax.set_xlabel(
    # ax.set_ylabel(
    # b1 = plt.Rectangle((basins[f'{lig}-bnd'][0]/10, basins[f'{lig}-bnd'][2]/10),

    # cbar.set_label('Free Energy (kcal/mol)') 


def snapshot_pdbs(directory, trj_path, top_path, snapshots, ref_str=None):
    # Make the directory for the output
    try:
        subprocess.run(f"mkdir -p {directory}/snapshots/",
                       shell=True, check=True)
    except subprocess.CalledProcessError as error:
        print('Error code:', error.returncode,
              '. Output:', error.output.decode("utf-8"))
    # Define the output name
    stem = trj_path.split('/')[-1].split('.')[0]
    if isinstance(snapshots[0], int):
        for ts in snapshots:
            try:
                subprocess.run(('echo 0 | gmx_mpi trjconv '
                                f"-f {trj_path} "
                                f"-s {top_path} "
                                f"-o {directory}/snapshots/{stem}_{ts}.pdb "
                                f"-dump {ts*1000}"),
                               shell=True, check=True)
            except subprocess.CalledProcessError as error:
                print('Error code:', error.returncode,
                      '. Output:', error.output.decode("utf-8"))
    elif isinstance(snapshots[0], list):
        for snl in snapshots:
            '''
            file1.append(f"trajin {trj_path} {' '.join([str(i) for i in snl])}")
            # Load reference structure (.top + .r)
            if isinstance(ref_str, list):
                file1.append(f"parm {ref_str[0]} [refparm]")
                file1.append(f"reference {ref_str[1]} parm [refparm]")
            # Load reference structure (.pdb)
            else:
                file1.append(f"reference {ref_str}")
            file1.append('rms reference @CA,C,N,O')
            file1.append(f"trajout {directory}/snapshots/{stem}.pdb multi keepext chainid A")
            file1.append("go")
            # Write all to cpptraj input file
            with open(f"{directory}/sn{snl[0]}.in", 'w') as file:
                file.writelines('\n'.join(file1))
            # Run cpptraj using that input file
            _run_cpptraj(directory, f"sn{snl[0]}.in")
            ONLY NECESSARY IF KEEPEXT NOT FUNCTIONAL
            for i in np.arange(len(snaps)):
                print(i, snaps[i])
                try:
                    subprocess.run(' '.join(['mv',
                                   f"{directory}/snapshots/{stem}.pdb.{i+1}",
                                   f"{directory}/snapshots/{stem}_{snaps[i]/200:.0f}ns.pdb"]),
                                   shell=True, check=True)
                except subprocess.CalledProcessError as error:
                    print('Error code:', error.returncode,
                          '. Output:', error.output.decode("utf-8"))
            '''


if __name__ == "__main__":

    ligand_res_names = {'A769': 'MOL',
                        'PF739': 'PF',
                        'SC4': 'SC',
                        'MT47': 'MOL',
                        'MK87': 'MOL'}

    i = 0
    for system in SYSTS:
        for lig in LIGS:
            for rep in ['R'+str(x) for x in np.arange(2)+1]:
            # for rep in ['R1']:
                # Define the working directory for each analysis
                wd = f"{DATA_DIR}/{system}+{lig}/06-MetaD/{rep}"
                print(system, lig, rep)

                '''
                # Create a final FES from the HILLS file
                run_sumhills(wd, f"{system}+{lig}")
                # Create a FES over time (every 250 ns)
                run_sumhills(wd, f"{system}+{lig}", stride=125000)
                '''

                '''
                # LIGAND & BACKBONE RMSD
                new_data = tt.measure_rmsd(f"{wd}/md_dry.pdb",
                    f"{wd}/metad_{system}+{lig}_final.xtc",
                    f"{wd}/md_dry.pdb",
                    [f"resname {ligand_res_names[lig]} and not name H*"]).run()

                # backbone:
                inp = pd.DataFrame(columns=['t', rep],
                                   data=new_data.results.rmsd[:, [1, 2]]).set_index('t')
                inp_l = pd.concat({lig: inp}, axis=1)
                inp_s = pd.concat({system: inp_l}, axis=1)

                # ligands:
                inp2 = pd.DataFrame(columns=['t', rep],
                                    data=new_data.results.rmsd[:, [1, 3]]).set_index('t')
                inp_l2 = pd.concat({lig: inp2}, axis=1)
                inp_s2 = pd.concat({system: inp_l2}, axis=1)

                if i == 0:
                    print('First time --> Creating Files')
                    inp_s.to_hdf(f"{DATA_DIR}/backbone_rmsd.h5", key='df')
                    inp_s2.to_hdf(f"{DATA_DIR}/ligand_rmsd.h5", key='df')
                    i += 1
                    continue
                print('Further time --> Reading Files & Adding Data')
                new = pd.read_hdf(f"{DATA_DIR}/backbone_rmsd.h5", key='df')
                new2 = pd.read_hdf(f"{DATA_DIR}/ligand_rmsd.h5", key='df')

                if any([(mi == inp_s.columns)[0] for mi in new.columns]):
                    print("Updating values in DataFrame.")
                    new.update(inp_s)
                else:
                    print("Adding new values to DataFrame.")
                    new = new.join(inp_s)
                # reorder columns
                new = new.iloc[:, new.columns.sortlevel(0, sort_remaining=True)[1]]
                new.to_hdf(f"{DATA_DIR}/backbone_rmsd.h5", key='df')

                if any([(mi == inp_s2.columns)[0] for mi in new2.columns]):
                    print("Updating values in DataFrame.")
                    new2.update(inp_s2)
                else:
                    print("Adding new values to DataFrame.")
                    new2 = new2.join(inp_s2)
                # reorder columns
                new2 = new2.iloc[:, new2.columns.sortlevel(0, sort_remaining=True)[1]]
                new2.to_hdf(f"{DATA_DIR}/ligand_rmsd.h5", key='df')

                ligand_atoms = {'A769': [43,44,45,46,47,48,49,52,58],
                                'PF739': [43,44,46,48,50,51,52,53,55],
                                'SC4': [38,39,40,41,42,43,44,45,46],
                                'MT47': [21,23,25,26,27,28,29,36,37,39],
                                'MK87': [24,25,26,27,28,29,30,31,32]}
                shift = -8 if system == 'a2b2' else 0
                ligand_core = f"{' or '.join([f'id {5900+x+shift}' for x in ligand_atoms[lig]])}"
                print(ligand_core)

                # OTHER RMSD --> H5 FILE
                # to_measure = [ligand_core]
                # to_align = 'backbone'
                # out_name = 'Ligand_Core'
                to_measure = ['backbone and resnum 273-365']
                to_align = 'backbone and resnum 1-272'
                out_name = 'Beta_AlphaAligned'
                new_data = tt.measure_rmsd(f"{wd}/md_dry.pdb",
                                           f"{wd}/metad_{system}+{lig}_final.xtc",
                                           f"{wd}/md_dry.pdb",
                                           to_measure,
                                           aln_group=to_align).run()

                # measured RMSD:
                inp2 = pd.DataFrame(columns=['t', rep+'-Alpha', rep+'-Beta'],
                        data=new_data.results.rmsd[:, 1:]).set_index('t')
                inp_l2 = pd.concat({lig: inp2}, axis=1)
                inp_s2 = pd.concat({system: inp_l2}, axis=1)

                if i == 0:
                    print('First time --> Creating File')
                    inp_s2.to_hdf(f"{DATA_DIR}/{out_name}_rmsd.h5", key='df')
                    i += 1
                    continue
                print('Further time --> Reading File & Adding Data')
                new2 = pd.read_hdf(f"{DATA_DIR}/{out_name}_rmsd.h5", key='df')

                if any([(mi == inp_s2.columns)[0] for mi in new2.columns]):
                    print("Updating values in DataFrame.")
                    new2.update(inp_s2)
                else:
                    print("Adding new values to DataFrame.")
                    new2 = new2.join(inp_s2)
                # reorder columns
                new2 = new2.iloc[:, new2.columns.sortlevel(0, sort_remaining=True)[1]]
                new2.to_hdf(f"{DATA_DIR}/{out_name}_rmsd.h5", key='df')

                new_strideplot(wd,
                            f"{system}+{lig}",
                            stride=250,
                            #                   proj       ext
                            basins={'bound':   [0.0, 1.0, 0.0, 0.75],
                                    'unbound': [3.5, 4.5, 0.0, 0.5]})
                '''

                gismo_traj(wd, f"metad_{system}+{lig}_final.xtc",
                           f"{system}+{lig}_{rep}_GISMO.xtc")
                gismo_colvar(wd,
                             out_colvar=f"{system}+{lig}_{rep}_GISMO.colvar")

    '''
    fes_by_replica(basins={'bound': [0.0, 1.0, 0.0, 0.75], 'unbound': [3.5, 4.5, 0.0, 0.5]})
    fes_by_system(basins={'bound': [0.0, 1.0, 0.0, 0.75], 'unbound': [3.5, 4.5, 0.0, 0.5]})
    ''' 
