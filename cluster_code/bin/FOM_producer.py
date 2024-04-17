import sys,os,os.path

sys.path.append("/gluster/data/next/software/IC_sophronia")   # cite IC from parent directory
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/gluster/data/next/software/IC_sophronia/IC'

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import tables as tb
import IC.invisible_cities.io.dst_io                           as     dstio
import IC.invisible_cities.io.mcinfo_io as mcio
from    IC.invisible_cities.core.core_functions   import shift_to_bin_centers
#import iminuit,probfit

import scipy.special as special
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

import fom_functions as func

def full_monty(path, port, output_folder):
    '''
    Will do everything as explained below

    path is directory
    port is port of interest (1a, 1b, 2a, 2b)
    output_folder is filepath and name of folder relative to path

    So for example, you have a folder structure of:
    105_7e-3/PORT_1a/isaura/isaura_1_208Tl.h5
    105_7e-3/PORT_1a/isaura/isaura_2_208Tl.h5
    105_7e-3/PORT_1a/isaura/...
    105_7e-3/PORT_1a/isaura/isaura_300_208Tl.h5
    .
    .
    .
    90_7e-3/PORT_1a/isaura/isaura_1_208Tl.h5
    90_7e-3/PORT_1a/isaura/isaura_2_208Tl.h5
    and so on.

    You input the path to the outer directory, and the port of interest
    and it will collect and process the isaura data within it.

    And output do a output folder respective to the path.
    '''
    ####################################
    # Change parameters for cutting here:
    ####################################

    # FIDUCIAL
    lower_z = 20
    upper_z = 1195
    r_lim = 472

    # ENERGY CUTS
    lower_e = 1.5
    upper_e = 1.7

    # SATELLITE REMOVAL
    energy_limit = 0.05

    ####################################
    # FOM cut list
    ####################################
    cut_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

    print("Opening files...")
    # load data from path
    dire = path + "PORT_" + str(port) + "/isaura/"
    try:
        data = func.load_data(dire)
    except:
        print("No data found for directory:\n{}\nSkipping...".format(dire))
        return 0

    tracks      = data[0]
    particles   = data[1]
    eventmap    = data[2]

    print("Applying cuts")

    # removing satellite tracks
    low_e_cut_tracks = func.remove_low_E_events(tracks, energy_limit)

    # apply cuts
    cut_output = func.apply_cuts(low_e_cut_tracks, lower_z, upper_z, r_lim, lower_e, upper_e)
    cut_data = cut_output[0]
    efficiencies = cut_output[1]

    print("Calculating FOM")

    # calculate FOM
    fom_output = func.apply_FOM(dire, cut_data, cut_list)

    # apply them to the efficiencies
    efficiencies.loc[len(efficiencies.index)] = ['pos_evt - all_evt', fom_output[0], len(cut_data), 0]
    efficiencies.loc[len(efficiencies.index)] = ['FOM_MAX - blob2_E_val (MeV)', fom_output[2], fom_output[3], 0]
    efficiencies.loc[len(efficiencies.index)] = ['trk_no - satellite_no', len(tracks.index), len(tracks.index) - len(low_e_cut_tracks.index), 0]

    # write to respective directories
    out_dir = path+output_folder
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    efficiencies.to_csv(str(out_dir) + '/efficiency.csv')
    # Save the data to a h5 file
    cut_data.to_hdf(str(out_dir) + '/post_cuts.h5', key='cut_data', mode = 'w')
    print("Data written")

    # fom output is POSITRON EVENTS, TOTAL EVENTS, MAX_FOM, blob val at max fom
    return (efficiencies)

def FOM_recalc(path, port, output_folder, full_output_name):
    # collect all the folders here
    try:
        file_names = [f for f in os.listdir(path)]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")

    monty_output = []
    # delete already existing table
    if os.path.exists(path + full_output_name):
        os.remove(path + full_output_name)

    for i in range(len(file_names)):
        path_ = path + str(file_names[i]) + "/"
        monty_output.append(full_monty(path_, str(port), str(output_folder)))
        monty_output[i].to_hdf(path + full_output_name, key=str(file_names[i]), mode = 'a', format='table', data_columns=True)




# set folder_path here!
if __name__ == '__main__':
    # make this the full path, as in the path to the folder with folders:
    #   105_7e-3/
    #   105_5e-3/
    #   105_3e-3/
    #   ...

    folder_path     = sys.argv[1]
    port            = sys.argv[2]
    output_folder   = sys.argv[3]
    full_output_name= sys.argv[4]
    print("Processing data at:\n{}".format(folder_path))
    FOM_recalc(str(folder_path), str(port), str(output_folder), str(full_output_name))
                                    
