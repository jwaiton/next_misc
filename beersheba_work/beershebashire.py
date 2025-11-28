'''
beershebashire - a crappy adage to beersheba that:
- rebins the data so that there aren't a stupid number of interpolated hits
- drops isolated hits

This is basically a 'soft' voxelisation before paolina functions are applied, but its a necessity for working with hits directly.

'''

import numpy as np
import pandas as pd
import os
import sys
import argparse
from tqdm import tqdm
import tables as tb

from invisible_cities.cities.beersheba import drop_isolated
from invisible_cities.io.dst_io        import df_writer
from invisible_cities.io.dst_io        import load_dst



def rebin(df, dx, dy, dz, q_type = 'Qc', e_type = 'E'):
    '''
    Docstring for rebin

    :param df: Description
    :param dx: Description
    :param dy: Description
    :param dz: Description
    '''

    x_edge = np.arange(df.X.min(), df.X.max() + dx, dx)
    y_edge = np.arange(df.Y.min(), df.Y.max() + dy, dy)
    z_edge = np.arange(df.Z.min(), df.Z.max() + dz, dz)

    # meshgrid from voxel centres
    x_centres = (x_edge[:-1] + x_edge[1:]) / 2
    y_centres = (y_edge[:-1] + y_edge[1:]) / 2
    z_centres = (z_edge[:-1] + z_edge[1:]) / 2

    xx, yy, zz = np.meshgrid(x_centres, y_centres, z_centres, indexing='ij')

    He, edges = np.histogramdd(
                            sample = np.vstack([df.X, df.Y, df.Z]).T,
                            bins   = [x_edge, y_edge, z_edge],
                            weights = df[e_type])
    #if 'Q' in df:
    #    Hq, edges = np.histogramdd(
    #                        sample = np.vstack([df.X, df.Y, df.Z]).T,
    #                        bins   = [x_edge, y_edge, z_edge],
    #                        weights = df[q_type])


    rebin_df = pd.DataFrame({
                     "event": df.event.unique()[0],
                     "npeak": df.npeak.unique()[0], # this should not just extract the first peak, very naughty
                     "X"        : xx.ravel(),
                     "Y"        : yy.ravel(),
                     "Z"        : zz.ravel(),
                     "E"        : He.ravel(),
                     "Xpeak"    : 0.0,
                     "Ypeak"    : 0.0, # we overwrite this garbage because we dont use it, naughty
                     "time"     : df.time.unique()[0], # this should be consistent across the whole event
                     #"Ec"       : He.ravel(),
                     #"Q"        : Hq.ravel(), # dodgy
                     #"Qc"       : Hq.ravel(),
                     #"track_id" : -1,
                     #"Ep"       : -1.0,
                     "Xrms"     : 0.0,
                     "Yrms"     : 0.0,
                     "nsipm"    : 1,
                     #"npeak"    : 0,
                     })

    # remove empty hits (you need to do this)
    rebin_df = rebin_df[rebin_df['E'] > 0].reset_index(drop=True)

    return rebin_df


def read_data(input, run_number, h5_pattern):
    '''
    takes in and reads out all the required beersheba tables
    '''
    dsts = {}

    for key, value in h5_pattern:
        try:
            dsts[key] = load_dst(input, value[0], value[1])
        except Exception as e:
            # out of laziness to avoid later crashing, insert empty df
            dsts[key] = pd.DataFrame()
            print(f'File {input} broke:\n{e}')

    # read out all this madness from a dictionary
    return dsts


def save_data(dsts, rebin_df, output_directory, save_file_name, run_number):
    '''
    saves individual files
    '''

    print(f'Saving data to {output_directory}{save_file_name}')
    with tb.open_file(f'{output_directory}{save_file_name}', 'w') as h5out:
        df_writer(h5out, dsts['DST'],      'DST',     'Events')
        df_writer(h5out, rebin_df,         'DECO',    'Events')
        df_writer(h5out, dsts['filters'],  'Filters', 'nohits')
        df_writer(h5out, dsts['runevts'],  'Run',     'events')
        df_writer(h5out, dsts['runinfo'],  'Run',     'runInfo')
        df_writer(h5out, dsts['conf'],     'config',  'beersheba')

        if int(run_number) <= 0:
            df_writer(h5out, dsts['MC_conf'],          'MC',  'configuration')
            df_writer(h5out, dsts['MC_evmp'],          'MC',  'event_mapping')
            df_writer(h5out, dsts['MC_hits'],          'MC',  'hits')
            df_writer(h5out, dsts['MC_particles'],     'MC',  'particles')
            df_writer(h5out, dsts['MC_snspos'],        'MC',  'sns_positions')
            df_writer(h5out, dsts['MC_sns_resp'],      'MC',  'sns_response')
            df_writer(h5out, dsts['MC_evtmp'],         'MC',  'eventMap')


def beershireba(input_directory, output_directory, run_number, timestamp, rebin_d = [5,5,4], drop_dist = [16, 16, 4], nhits = 3):

    # ensure paths exist, if not either error out or make one
    if not os.path.isdir(input_directory):
        raise FileNotFoundError(f'Directory does not exist: {input_directory}')
    if not os.path.isdir(output_directory):
        print(f'{output_directory} does not exist, creating it now...')
        os.makedirs(output_directory, exist_ok=True) # no need for the exist_ok check, but left in

    # save time in the loading: dictionary name,  h5 group,  h5 node
    h5_pattern = {'DST'      :  ['DST',     'Events'],
                  'rebin_df' :  ['DECO',    'Events'],
                  'filters'  :  ['Filters', 'nohits'],
                  'runevts'  :  ['Run',     'events'],
                  'runinfo'  :  ['Run',     'runInfo'],
                  'conf'     :  ['config',  'beersheba']}

    # adding MC
    if int(run_number) <= 0:
        h5_pattern['MC_conf']      = ['MC',  'configuration']
        h5_pattern['MC_evmp']      = ['MC',  'event_mapping']
        h5_pattern['MC_hits']      = ['MC',  'hits']
        h5_pattern['MC_particles'] = ['MC',  'particles']
        h5_pattern['MC_snspos']    = ['MC',  'sns_positions']
        h5_pattern['MC_sns_resp']  = ['MC',  'sns_response']
        h5_pattern['MC_evtmp']     = ['MC',  'eventMap']





    drop_sensors = drop_isolated(drop_dist, ['E'], nhits)
    # extract all files in the input directory
    files = [f for f in os.listdir(input_directory) if f.endswith('.h5')]
    for f in tqdm(files):

        # names always have 4 digits in them in this land, lets extrapolate that
        # and use it for the output
        evt_num = next(dig for dig in f.split('_') if dig.isdigit() and len(dig) == 4)
        LDC_num = next((p for p in f.split('_') if "ldc" in p.lower()), None)
        output_name = f'run_{run_number}_{evt_num}_{LDC_num}_{timestamp}_beershireba.h5'


        dsts = read_data(f'{input_directory}{f}', run_number, h5_pattern)

        file_data = []
        for i, df in tqdm(dsts['DECO'].groupby('event')):
            print('=' * 20)
            print(f'event {i}:')
            print('=' * 20)

            # rebin
            print(f'rebinning with {rebin_d[0]}, {rebin_d[1]}, {rebin_d[2]}')
            rebinned_df = rebin(df, rebin_d[0], rebin_d[1], rebin_d[2])
            print(f'rebinned from {df.shape} to {rebinned_df.shape}')
            # drop isolated sensors
            print('dropping...')
            dropped_df  = drop_sensors(rebinned_df.copy())
            # append to the lsit
            file_data.append(dropped_df)

        new_df = pd.concat(file_data)
        # save
        print('saving...' )
        save_data(dsts, new_df, output_directory, output_name, run_number)



def arg_parser():
    parser = argparse.ArgumentParser(
                            description = "Rebins and drops isolated sensors post beersheba."
                            )

    # required
    parser.add_argument(
            'input_directory',
            type = str,
            help = "Input directory path (relative or absolute)",
            )

    parser.add_argument(
            'output_directory',
            type = str,
            help = "Output directory path (relative or absolute)",
            )


    parser.add_argument(
            'run_number',
            type = int,
            help = "Run number, -1 if MC",
            )

    parser.add_argument(
            'timestamp',
            type = int,
            help = "Timestamp",
            )

    parser.add_argument(
            '--rebin_d',
            type    = float,
            nargs   =3,
            help    = "rebinning dimenstions, dx, dy, dz",
            default = [5, 5, 4]
            )

    parser.add_argument(
            '--drop_dist',
            type    = float,
            nargs   = 3,
            help    = "drop distance dimsnesions, dx, dy, dz",
            default = [16, 16, 4]
            )

    parser.add_argument(
            '--nhits',
            type    = int,
            help    = "Number of hits required to be considered a non-isolated cluster",
            default = 3
            )

    args = parser.parse_args()
    unpacked_args = {k: v for k, v in vars(args).items() if v is not None}

    return unpacked_args


if __name__ == '__main__':
    args = arg_parser()

    for key, value in args.items():
        print(f'{key}: {value}')

    beershireba(**args)
