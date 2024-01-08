import sys,os,os.path

sys.path.append("../../")   # cite IC from parent directory
                            # NOTE if you can't import IC stuff, its because of the
                            # above line
#sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/home/e78368jw/Documents/NEXT_CODE/IC'

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import tables as tb
import IC.invisible_cities.io.dst_io                           as     dstio
import IC.invisible_cities.io.mcinfo_io as mcio
from    IC.invisible_cities.core.core_functions   import shift_to_bin_centers
import iminuit,probfit

import scipy.special as special
from scipy.stats import skewnorm
from scipy.optimize import curve_fit



def positron_scraper(data_path, save = False):
    """
    Function that iterates over files with MC and collects only positron events.
    Intended to reduce the memory resources of MC data.
    """



     # collect all filenames
    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")


    # read in a singular file to collect the column titles
    
    MC_df_single = pd.read_hdf(data_path + file_names[0], 'MC/particles')

    MC_df = []
    pos_df = pd.DataFrame(columns = MC_df_single.columns)
    eventmap = []


    i = 0
    

    # how much you chunk your data
    chunker = np.floor(len(file_names)*0.1)

    # chunk file_names
    
    for file in file_names:
        file_path = data_path + file

        # load in file
        MC_df_temp = pd.read_hdf(file_path, 'MC/particles')
        MC_df.append(MC_df_temp)
        eventmap.append(mcio.load_eventnumbermap(file_path).set_index('nexus_evt'))


        i += 1

        # chunk checker, every time you hit a certain chunk,
        # collect the positron events and wipe the df
        if ((i%chunker) == 0):
            print("Chunking at event {}!".format(i))
            # concat the list
            MC_df = pd.concat(MC_df, axis = 0, ignore_index = True)
            #print("Post concat")
            #display(MC_df)
            pos_data = MC_df[MC_df['particle_name'] == 'e+']

            
            #display(pos_data)
            #print(type(pos_data))
            # collect positron events into df
            pos_df = pos_df.append(pos_data)
            print("{} positron events found\n{} positron events total".format(pos_data.shape[0],pos_df.shape[0]))
            #display(pos_df)

            # make space
            MC_df = []

    if (save == True):
        pos_df.to_hdf('positrons.h5', key = 'pos', mode = 'w')

    return pos_df




def load_MC(data_path):
    """
    Load in MC data


    Returns eventmap and particles together as tuple
    """

     # collect all filenames
    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")


    # NOTE Break this section up, its annoying like this.
    df_trs = []
    df_ems = []
    # create massive dataframe with all of them
    for file in file_names:
        file_path = data_path + file
        # include MC particles (boooo takes ages)
        # collecting the correct components of the file, not exactly sure how this works
        df_ps = pd.read_hdf(file_path, 'MC/particles')
        #df_ps = df_ps[df_ps.creator_proc == 'conv']
        # collecting event map
        df_em = mcio.load_eventnumbermap(file_path).set_index('nexus_evt')
        df_trs.append(df_ps)
        df_ems.append(df_em)

    particles = pd.concat(df_trs, ignore_index=True)
    eventmap = pd.concat([dt for dt in df_ems])

    return (eventmap, particles)




def load_tracks(data_path, filter = 0):
    """
    Load in tracks from the isaura data. Applies filters iteratively
    to allow for larger file manipulation
    
    Filter: 0 -> no filter
            1 -> one-track filter
    """
    

    # collect all filenames
    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")

    # create list to hold dataframe
    df_tracks = []
    f_tracks = []
    # if you dont want to do the filtering, run here

    # load data
    q = 0
    r = len(file_names)
    print("Warning! This method may take some time,\nand works best with smaller datasets (like LPR).")
    for file in file_names:
        
        file_path = data_path + file
        df = dstio.load_dst(file_path, 'Tracking', 'Tracks')
        df_tracks.append(df)

        # tracking
        q += 1
        parsing = ((q//r)*100)
        if (parsing%10 == 0) and (parsing != 0):
            print("{:.2f} %".format(parsing))

        # if filtering
        if (filter != 0) and (q%round(r/10)== 0):
            # concat
            tracks = pd.concat(df_tracks, axis = 0, ignore_index = True)
            # remove events with more than one track
            f_tracks.append(tracks[tracks['numb_of_tracks'] == 1])
            # remove df_tracks
            df_tracks = []


    
    # concat the list
    if (filter == 0):
        tracks = pd.concat(df_tracks, axis = 0, ignore_index = True)
    else:
        tracks = pd.concat(f_tracks, axis = 0, ignore_index = True)
    
    return tracks




#def FOM_calc(data_path, title = 'FOM plot'):

