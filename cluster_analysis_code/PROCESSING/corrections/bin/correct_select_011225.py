import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tables as tb
from matplotlib import colors
from tqdm import tqdm
import json



from typing          import Optional
from typing          import Union
from typing          import Callable
from typing          import Any

from concurrent.futures import ProcessPoolExecutor

import sys,os,os.path
from pathlib import Path
sys.path.append("/scratch/halmazan/NEXT/IC_include-cluster-dropping/IC/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
sys.path.append("/scratch/halmazan/NEXT/testing/notebooks/")
os.environ['ICTDIR']='/scratch/halmazan/NEXT/IC_include-cluster-dropping/'

from invisible_cities.io.dst_io           import load_dst, load_dsts, df_writer
from invisible_cities.io.hits_io          import hits_writer
from invisible_cities.core                import tbl_functions   as tbl
from invisible_cities.core.core_functions import in_range

from invisible_cities.io.hits_io          import hits_from_df
from invisible_cities.evm.nh5             import HitsTable
from invisible_cities.types.symbols       import NormStrategy
from invisible_cities.types.ic_types      import NoneType
from invisible_cities.reco.corrections    import read_maps, get_df_to_z_converter, apply_all_correction
from invisible_cities.evm.event_model     import HitCollection

import glob
from tqdm import tqdm


##################################################################

def identity(x : Any) -> Any:
    return x


def collect_maps(  map_path   : str
                 , apply_temp : bool
                 , norm_strat : NormStrategy):

    maps     = read_maps(os.path.expandvars(map_path))
    get_coef = apply_all_correction( maps
                                   , apply_temp = apply_temp
                                   , norm_strat = norm_strat)

    return get_coef


def load_all(f):

    dst     = load_dst(f, 'DST',  'Events')
    hits    = load_dst(f, 'RECO', 'Events')
    events  = load_dst(f, 'Run',  'events')
    runInfo = load_dst(f, 'Run',  'runInfo')

    return dst, hits, events, runInfo


def get_calibration_constants(R):
    # read in the runs gradient and intercept values
    json_path = f'/scratch/halmazan/NEXT/PROCESSING/corrections/cor_map/corrections.json'
    with open(json_path, 'r') as file:
        corrections = json.load(file).get(R, {})

    return corrections.get('M', None), corrections.get('C', None)


def correct_martin(df, R):
    '''
    Read in the json stored wherever I put it and apply the corrections to energy.
    For each differing peak, correct energy wrt total
    '''

    df = df.copy()

    m, c = get_calibration_constants(R)

    ###df['Ec'] = df.Ec * m + c/len(df)

    ec_tot = df.Ec.sum()
    ec_cor = (ec_tot * m) + c
    ##ec_diff = ec_cor - ec_tot

    ##df['Ec'] += ec_diff * (df['Ec']/ec_tot)
    df['Ec'] = df['Ec'] * (ec_cor/ec_tot)

    return df


def correct_and_cut(  R   : int
                    , TS  : int
                    , MAP : str
                    , LDC : int):
    '''
    Collect the relevant sophronia files, correct the energy and cut around the DEP

    R   -> run number
    TS  -> timestamp
    MAP -> correction map
    '''

    compression = 'ZLIB4'

    # setup coef
    get_coef = collect_maps( MAP
                           , False
                           , NormStrategy.kr)

    # file in and out direction
    files = f'/data/halmazan/NEXT/N100_LPR/{R}/sophronia/prod/ldc{LDC}/'
    print(f'INPUT: {files}')
    files = sorted(glob.glob(files + '*'), key=lambda x: (x.split('/')[-2],int(x.split('/')[-1].split('_')[2])))
    file_out = f'/scratch/halmazan/NEXT/N100_LPR/{R}/sophronia/{TS}/ldc{LDC}'

    print(f'Making output directory @ {file_out}')

    # make sure the end directory exists
    os.makedirs(file_out, exist_ok=True)

    passing_dsts = []
    passing_hits = []
    passing_evts = []
    passing_rInf = []

    for i, f in enumerate(files):

        R, I = files[i].split('/')[-1].split('_')[1:3]

        # load in
        try:
            dst, hits, events, runInfo = load_all(f)
        except Exception as e:
            print(f'Error loading file: {e}')
            continue

        if hits.empty:
            print(f'Empty file, skipping: {f}')
            continue

        # correct
        get_all_coefs = get_coef(hits.X.values, hits.Y.values, hits.Z.values, hits.time.values)
        hits['Ec']    = hits.E * get_all_coefs

        # take only events with corrected energy in the range
        hits = hits[hits.groupby('event').Ec.transform('sum') < 0.4]
        hits = hits[hits.groupby('event').Ec.transform('sum') > 0.7]

        # apply martins correction
        hits = hits.groupby('event', group_keys = False).apply(lambda group: correct_martin(group, R=R))

        valid_events = hits.event.unique()
        # only keep events in the other dataframes that match
        dst     = dst[dst['event'].isin(valid_events)]
        events  = events[events['evt_number'].isin(valid_events)]
        runInfo = runInfo.head(len(valid_events))

        with tb.open_file(f'{file_out}/run_{R}_{I}_ldc{LDC}_{TS}.h5', 'w', filters = tbl.filters(compression)) as h5out:
            df_writer(h5out,   dst, "DST", "Events" , compression="ZLIB4")
            df_writer(h5out,   hits, "RECO", "Events" , compression="ZLIB4")
            df_writer(h5out,   runInfo, "Run" , "runInfo" , compression="ZLIB4")
            df_writer(h5out,   events, "Run" , "events", compression="ZLIB4")

        #passing_dsts.append(dst)
        #passing_hits.append(hits)
        #passing_evts.append(events)
        #passing_rInf.append(runInfo)

    #passing_dsts = pd.concat(passing_dsts)
    #passing_hits = pd.concat(passing_hits)
    #passing_evts = pd.concat(passing_evts)
    #passing_rInf = pd.concat(passing_rInf)





if __name__ == '__main__':
    R      = sys.argv[1]
    TS     = sys.argv[2]
    M_PATH = sys.argv[3]
    LDC    = sys.argv[4]
    print('starting up python program...')
    correct_and_cut(R, TS, M_PATH, LDC)
