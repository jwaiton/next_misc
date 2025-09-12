import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
from tqdm import tqdm
import tables as tb
from matplotlib import colors 

import sys,os,os.path

from invisible_cities.io.dst_io           import load_dst
from invisible_cities.io.dst_io           import df_writer


def rebin(input_path, output_path, rebin_value):
    DST   = load_dst(input_path, 'DST', 'Events')
    RECO  = load_dst(input_path, 'RECO', 'Events')
    evt  = load_dst(input_path, 'Run', 'events')
    rinfo = load_dst(input_path, 'Run', 'runInfo')
    fils  = load_dst(input_path, 'Filters', 's12_selector')
    vldh  = load_dst(input_path, 'Filters', 'valid_hit')
    DBPM  = load_dst(input_path, 'DB', 'DataPMT')
    DBSi  = load_dst(input_path, 'DB', 'DataSiPM')

   
    rebin_value = int(rebin_value) 
    rebinned_RECO = []
    # rebin
    for evts, df in RECO.groupby('event'):
        zbin = (df.Z // rebin_value) * rebin_value
        df['Z'] = zbin
        q = df.groupby('event time npeak nsipm X Y Xrms Yrms Z track_id Ep'.split()).agg(dict(E="sum", Q="sum", Ec="sum", Qc="sum")).reset_index()
        rebinned_RECO.append(q)
    
    rebinned_RECO = pd.concat(rebinned_RECO)

    # write
    with tb.open_file(output_path, 'w') as h5out:
        df_writer(h5out, DST, 'DST', 'Events')
        df_writer(h5out, rebinned_RECO, 'RECO', 'Events')
        df_writer(h5out, evt, 'Run', 'events')
        df_writer(h5out, rinfo, 'Run', 'runInfo')
        df_writer(h5out, fils, 'Filters', 's12_selector')
        df_writer(h5out, vldh, 'Filters', 'valid_hit')
        df_writer(h5out, DBPM, 'DB', 'DataPMT')
        df_writer(h5out, DBSi, 'DB', 'DataSiPM')

input_path = sys.argv[1]
print(f'Input: {input_path}')
output_path = sys.argv[2]
print(f'Output: {output_path}')
rebin_value = sys.argv[3]
print(f'Rebinning with factor {rebin_value}')
print('This should correspond to double/triple/n-le the spacing between Z positions.')

rebin(input_path, output_path, rebin_value)
