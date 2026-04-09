import sys,os,os.path
import csv
import traceback
#sys.path.append("../../")   # cite IC from parent directory
sys.path.append("/gluster/data/next/software/IC_311024/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
sys.path.append("/gluster/data/next/notebooks/")
os.environ['ICTDIR']='/gluster/data/next/software/IC_311024/'

from invisible_cities.core.core_functions   import shift_to_bin_centers
from invisible_cities.io.dst_io           import load_dst, load_dsts, df_writer

import FOM_functions as FOM_func
import functions_HE as func
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.special as special
from scipy.stats import norm
from scipy.stats import skewnorm, crystalball
from scipy.optimize import curve_fit
from tqdm import tqdm

from scipy.integrate import quad

import iminuit
from iminuit import Minuit
import probfit
from concurrent.futures import ProcessPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tables as tb
from matplotlib import colors

from typing          import Optional
from typing          import Union
from typing          import Callable

from concurrent.futures import ProcessPoolExecutor

import sys,os,os.path
from pathlib import Path

from invisible_cities.io.dst_io           import load_dst, load_dsts, df_writer
from invisible_cities.io.hits_io          import hits_writer
from invisible_cities.core                import tbl_functions   as tbl
from invisible_cities.core.core_functions import in_range
#from invisible_cities.cities.beersheba    import hitc_to_df_
from invisible_cities.io.hits_io          import hits_from_df
from invisible_cities.evm.nh5             import HitsTable
from invisible_cities.types.symbols       import NormStrategy
from invisible_cities.types.ic_types      import NoneType
from invisible_cities.reco.corrections    import read_maps, get_df_to_z_converter, apply_all_correction
from invisible_cities.evm.event_model     import HitCollection

from tqdm import tqdm

import traceback


def load_in_data(FOM_TS):
    # load in the data
    CITY = 'isaura'
    #FOM_TS = ['200326_15']
    #FOM_TS = ['201025']
    TIMESTAMP = FOM_TS
    RUN_NUMBER = 'th_port1a_dep_202602_subsample'
    # make directory
    pre_dir = '/gluster/data/next/files/TOPOLOGY_John/MC_data/'
    folder_name = f'{pre_dir}/{RUN_NUMBER}/{CITY}/{FOM_TS[0]}'
    folder_s = Path(f'{folder_name}')

    print(folder_s)

    tdst = []
    for LDC in tqdm(range(1,8)):
        files = list(Path(f'{folder_name}/ldc{LDC}/').rglob('*.h5'))
        for file in tqdm(files):
            try:
                df = load_dst(file, 'Tracking', 'Tracks')

                lower_E = 1.5
                upper_E = 1.7

                tdst_sum_pE = df.groupby('event').energy.sum()
                tdst_sum_pE_cut = tdst_sum_pE[(tdst_sum_pE.values >= lower_E) & (tdst_sum_pE.values <= upper_E)]
                ROI_tdst = df[df.event.isin(tdst_sum_pE_cut.index)]

            except:
                traceback.format_exc()
                ROI_tdst = pd.DataFrame()
            tdst.append(ROI_tdst)

    tdst = pd.concat(tdst)
    return tdst

# summary
def make_summary(df, label = ''):
    print('=' * 20)
    print(f'Events: {df.event.nunique()}')
    plt.hist2d(df.numb_of_tracks, df.energy, bins = (20, 50))
    plt.xlabel('Number of tracks')
    plt.ylabel('Energy (MeV)')
    plt.savefig('/gluster/data/next/notebooks/john_books/year_of_horse_notebooks/beer_thekla_resolution/number_of_tracks_{label}.png')
    plt.show()
    plt.hist(df.groupby('event').energy.sum())
    plt.xlabel('Energy (MeV)')
    plt.ylabel('counts')
    plt.savefig('/gluster/data/next/notebooks/john_books/year_of_horse_notebooks/beer_thekla_resolution/energy_dist{label}.png')
    plt.show()
    print('=' * 20)


def visualise_tracks(df,
                     lower_E = 1.5,
                     upper_E = 1.7,
                     r_lim   = 300,
                     lower_z = 20,
                     upper_z = 1170,
                     label = ''):

    make_summary(df)
    # cut energy-wise
    tdst_sum_pE = df.groupby('event').energy.sum()
    tdst_sum_pE_cut = tdst_sum_pE[(tdst_sum_pE.values >= lower_E) & (tdst_sum_pE.values <= upper_E)]
    print(tdst_sum_pE_cut)
    ROI_tdst = df[df.event.isin(tdst_sum_pE_cut.index)]
    print('Post energy cut:')
    make_summary(ROI_tdst)

    print(f'Number of 1-tracks after ROI cuts: {ROI_tdst[ROI_tdst.numb_of_tracks == 1].event.nunique()}')

    fid_tdst = func.fiducial_track_cut_2(ROI_tdst, r_lim, lower_z, upper_z)
    print('Post fiducial cut:')
    make_summary(fid_tdst)

    # plot the number of tracks against energy
    plt.hist2d(fid_tdst.numb_of_tracks, fid_tdst.energy, bins = (20, 50), range = ([0, 50], [0, 1.7]))
    plt.xlabel('Number of tracks')
    plt.ylabel('Energy (MeV)')
    plt.savefig('/gluster/data/next/notebooks/john_books/year_of_horse_notebooks/beer_thekla_resolution/number_of_tracks_post_fid_{label}.png')
    plt.show()

    # plot the number of tracks against energy
    plt.hist2d(fid_tdst.numb_of_tracks, fid_tdst.energy, bins = (20, 50), range = ([0, 5], [0, 1.7]))
    plt.xlabel('Number of tracks')
    plt.ylabel('Energy (MeV)')
    plt.savefig('/gluster/data/next/notebooks/john_books/year_of_horse_notebooks/beer_thekla_resolution/energy_hist_post_fid_{label}.png')
    plt.show()


    print(f'Number of 1-tracks after all cuts: {fid_tdst[fid_tdst.numb_of_tracks == 1].event.nunique()}')



def main():

    data_type = '250326_12'
    tdst = load_in_data([data_type])
    visualise_tracks(tdst, label = data_type)

if __name__ == "__main__":
    main()
