import sys,os,os.path
import numpy as np
import pandas as pd
from tqdm import tqdm
import tables as tb

sys.path.append("/scratch/halmazan/NEXT/IC_alter-blob-centre/IC/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/scratch/halmazan/NEXT/IC_alter-blob-centre/IC/'

from invisible_cities.io.dst_io           import load_dst, load_dsts, df_writer
from invisible_cities.io.hits_io        import hits_from_df
from invisible_cities.cities.components import track_blob_info_creator_extractor
from invisible_cities.core              import system_of_units as units

def extract_MC_tracking(input_path, output_path):
    '''
    does what it says on the tin, extracts the MC tracking information
    '''


    # extract and reshape the df for topological info extraction
    hits_dst = load_dst(input_path, 'MC', 'hits')
    hits_dst.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z', 'energy': 'E', 'event_id' : 'event'}, inplace=True)
    hits_dst['npeak'] = 1
    reshaped_hits = hits_dst.rename(columns={'E': 'Ep'}).assign(Q=1)[['event', 'npeak', 'X', 'Y', 'Z', 'Q', 'Ep']]
    reshaped_hits['E'] = reshaped_hits.Ep
    hits_lol = hits_from_df(reshaped_hits)

    # topological creation
    topological_creator = track_blob_info_creator_extractor((18 * units.mm, 18 * units.mm, 18 * units.mm),
                                                        False,
                                                        0 * units.keV,
                                                        0,
                                                        45 * units.mm,
                                                        10000000000,
                                                        #scan_radius = 27 * units.mm
                                                        )


    full_df = []
    for evt in hits_lol.keys():
        df, track_hitc, out_of_map = topological_creator(hits_lol[evt])
        full_df.append(df)
    
    # concatenate
    full_df = pd.concat(full_df)

    with tb.open_file(output_path, 'w') as h5out:
        df_writer(h5out, full_df, 'Tracking', 'Tracks')


input_path = sys.argv[1]
print(f'Input: {input_path}')
output_path = sys.argv[2]
print(f'Output: {output_path}')


extract_MC_tracking(input_path, output_path)

