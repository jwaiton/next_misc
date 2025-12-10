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
from invisible_cities.cities.components   import copy_mc_info



def write_nonsense(output_path, number):
    '''
    write random h5 data to a file
    '''

    # generate some nonsense df
    data = {
        'A': np.random.randint(0, 100, 10),
        'B': np.random.random(10),
        'C': np.random.choice(['X', 'Y', 'Z'], 10)
    }
    DST = pd.DataFrame(data)


    full_path = f'{output_path}/file_{number}.h5'

    with tb.open_file(full_path, 'w') as h5out:
        df_writer(h5out, DST, 'DST', 'Events')

output_path = sys.argv[1]
num         = sys.argv[2]

write_nonsense(output_path, num)