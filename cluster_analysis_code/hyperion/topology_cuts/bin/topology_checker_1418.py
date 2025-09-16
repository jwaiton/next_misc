#########################################################
###
###
### Script that reads in data based on the RUN NUMBER and
### TIMESTAMP fed to it, then applies topological cuts,
### spits out an efficiency and the resulting dst.
###
###
#########################################################

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
sys.path.append("/scratch/halmazan/NEXT/IC_alter-blob-centre/IC/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
sys.path.append("/scratch/halmazan/NEXT/testing/notebooks/")
os.environ['ICTDIR']='/scratch/halmazan/NEXT/IC_alter-blob-centre/IC/'

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


import functions.functions_HE as func

############################################################

def load_single_file(file_path):
    '''
    Load data from a single h5 file and produce dataframes for /Tracking/Tracks

    Args:
        file_path       :       str
                                Path to the h5 file to be loaded.

    Returns:
        tracks_df       :       pandas.DataFrame
                                DataFrame containing the /Tracking/Tracks data.
        failed          :       int
                                1 if the file failed to load, 0 otherwise.
    '''
    try: 
        tracks_df = load_dst(file_path, 'Tracking', 'Tracks')
        return tracks_df, 0
    except Exception as e:
        print(f'File {file_path} broke with error:\n{e}')
        x = pd.DataFrame()
        return x, 1

def load_data_fast(folder_path):
    '''
    Load multiple h5 files and produce concatenated dataframes for /Tracking/Tracks, /MC/Particles, and their corresponding eventmap.

    Args:
        folder_path     :       str
                                Path to the folder containing the h5 files.

    Returns:
        tracks          :       pandas.DataFrame
                                Concatenated DataFrame containing the /Tracking/Tracks data from all h5 files.
        total_failures  :       int
                                Total number of failed file loads.
    '''
    
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    file_paths = [os.path.join(folder_path, f) for f in file_names]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(load_single_file, file_paths))
    
    # Separate the results into respective lists
    tracks_list, failures = zip(*results)

    tracks = pd.concat(tracks_list, axis=0, ignore_index=True)

    # Sum up the failures
    total_failures = sum(failures)

    return tracks, total_failures


def cut_and_save( R       : int
                , TS      : int
                , z_lower : float
                , z_upper : float
                , r_lim   : float
                , e_lower : float
                , e_upper : float
                , city    : str):


    print(f'R-{R}, TS-{TS}')
    
    root_path_data = '/data/halmazan/NEXT/'
    root_path = '/scratch/halmazan/NEXT/'

    # get a directory to save to
    folder_name = f'{root_path}PROCESSING/topology_cuts/data/{R}/{TS}/'
    folder_s = Path(f'{folder_name}')
    folder_s.mkdir(parents=True, exist_ok=True)

    # load in
    n100_dir = f'{root_path_data}N100_LPR/{R}/{city}/{TS}/'

    hdst = []
    errors = 0
    ldc_errors = 0
    for i in tqdm(range(1,8)):
        print(f"LDC {i}")
        folder_path = n100_dir + 'ldc' + str(i) + '/'
        try:
            holder, err = load_data_fast(folder_path)
            r = holder
            errors += err
            hdst.append(r)
        except Exception as e:
            print(f'{folder_path} broke! Probably because the entire LDC is broken if not told otherwise')
            print(f'{e}')
            ldc_errors += 1
    try:
        hdst = pd.concat(hdst)
    except:
        print('concat broke, who knows man')
    
    print('=' * 20)
    print(f'Number of failed files: {errors}')
    cut_hdst, efficiencies = func.apply_cuts(hdst, 
                                             lower_z = z_lower, 
                                             upper_z = z_upper, 
                                             r_lim   = r_lim, 
                                             lower_e = e_lower, 
                                             upper_e = e_upper)
    print('=' * 20)
    print(efficiencies)

    # SAVE THE DATAFRAME
    cut_hdst.to_hdf(f'{folder_name}cut_hdst_1418.h5', key = 'Tracking/Tracks')

if __name__ == '__main__':
    R       = sys.argv[1]
    TS      = sys.argv[2]
    z_lower = float(sys.argv[3])
    z_upper = float(sys.argv[4])
    r_lim   = float(sys.argv[5])
    e_lower = float(sys.argv[6])
    e_upper = float(sys.argv[7])
    city    = sys.argv[8]
    

    cut_and_save(R, TS, z_lower, z_upper, r_lim, e_lower, e_upper, city)
