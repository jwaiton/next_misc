import sys,os,os.path

sys.path.append("/gluster/data/next/software/IC_satkill")   # cite IC from parent directory
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/gluster/data/next/software/IC_satkill/IC'

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


import IC.invisible_cities.core.core_functions                   as     coref
import IC.invisible_cities.io.dst_io                           as     dstio

from IC.invisible_cities.cities                 import beersheba as beerfun

from IC.invisible_cities.evm.event_model                          import HitCollection

from IC.invisible_cities.database.load_db       import DataSiPM

from IC.invisible_cities.evm.event_model        import Cluster, Hit
from IC.invisible_cities.types.ic_types         import xy
from IC.invisible_cities.reco.paolina_functions import voxelize_hits, drop_end_point_voxels, make_track_graphs, get_track_energy

from IC.invisible_cities.evm.event_model        import HitEnergy
from IC.invisible_cities.cities.beersheba          import DeconvolutionMode
from IC.invisible_cities.cities.beersheba          import CutType

from IC.invisible_cities.reco import hits_functions as hif

from IC.invisible_cities.reco.deconv_functions import deconvolve
from IC.invisible_cities.reco.deconv_functions import deconvolution_input
from IC.invisible_cities.reco.deconv_functions import InterpolationMethod

import IC.invisible_cities.io.mcinfo_io as mcio

from collections import defaultdict


###
    #### THIS ALL ABOVE WILL HAVE TO BE CHANGED WRT WHERE WE ARE IN CLUSTER
###

###########################################################################################
###########################################################################################
######### DEFINE FUNCTIONS BELOW:
###########################################################################################
###########################################################################################




def threshold_hits(threshold_charge, same_peak, hitc):
    """
    Applies a threshold to hits and redistributes the charge/energy.

    Parameters
    ----------
    threshold_charge : float
        minimum pes of a hit
    same_peak        : bool
        whether to reassign NN hits' energy only to the hits from the same peak

    Returns
    ----------
    A function that takes HitCollection as input and returns another object with
    only non NN hits of charge above threshold_charge.
    The energy of NN hits is redistributed among neighbors.
    """

    t = hitc.time
    thr_hits = hif.threshold_hits(hitc.hits, threshold_charge     )
    mrg_hits = hif.merge_NN_hits ( thr_hits, same_peak = same_peak)

    cor_hits = []
    for hit in mrg_hits:
        cluster = Cluster(hit.Q, xy(hit.X, hit.Y), hit.var, hit.nsipm)
        xypos   = xy(hit.Xpeak, hit.Ypeak)
        hit     = Hit(hit.npeak, cluster, hit.Z, hit.E, xypos, hit.Ec)
        cor_hits.append(hit)

    new_hitc      = HitCollection(hitc.event, t)
    new_hitc.hits = cor_hits
    return new_hitc





def hits_from_df (dst, skip_NN = False):
    """
    Function that transforms pandas DataFrame dst to HitCollection
    ------
    Parameters
    ------
    dst : pd.DataFrame
        DataFrame with obligatory columns :
                event, npeak, X, Y, Z,  Q, E
        If time, nsipm, Xrms, Yrms, Qc, Ec, track_id are not
        inside dst the default value is set to -1
        If Xpeak, Ypeak not in dst the default value is -1000
    ------
    Returns
    ------
    Dictionary {event_number : HitCollection}
    from here
    https://github.com/next-exp/IC/blob/v2-development/invisible_cities/io/hits_io.py#L16
    """
    all_events = {}
    times = getattr(dst, 'time', [-1]*len(dst))
    for (event, time) , df in dst.groupby(['event', times]):
        #pandas is not consistent with numpy dtypes so we have to change it by hand
        event = np.int32(event)
        hits  = []
        for i, row in df.iterrows():
            Q = getattr(row,'Q', row.E)
            if skip_NN and Q == NN:
                continue
            if hasattr(row, 'Xrms'):
                Xrms  = row.Xrms
                Xrms2 = Xrms**2
            else:
                Xrms = Xrms2 = -1
            if hasattr(row, 'Yrms'):
                Yrms  = row.Yrms
                Yrms2 = Yrms**2
            else:
                Yrms = Yrms2 = -1
            nsipm   = getattr(row, 'nsipm'   , -1   )     # for backwards compatibility
            Qc      = getattr(row, 'Qc'      , -1   )     # for backwards compatibility
            Xpeak   = getattr(row, 'Xpeak'   , -1000)     # for backwards compatibility
            Ypeak   = getattr(row, 'Ypeak'   , -1000)     # for backwards compatibility
            Ec      = getattr(row, 'Ec'      , -1   )     # for backwards compatibility
            trackID = getattr(row, 'track_id', -1   )     # for backwards compatibility
            Ep      = getattr(row, "Ep"      , -1   )     # for backwards compatibility

            hit = Hit(row.npeak            ,
                      Cluster(Q               ,
                              xy(row.X, row.Y),
                              xy(Xrms2, Yrms2),
                              nsipm = nsipm   ,
                              z     = row.Z   ,
                              E     = row.E   ,
                              Qc    = Qc      ),
                      row.Z                ,
                      row.E                ,
                      xy(Xpeak, Ypeak)     ,
                      s2_energy_c = Ec     ,
                      track_id    = trackID,
                      Ep          = Ep     )

            hits.append(hit)

        if len(hits):
            all_events[event] = HitCollection(event, time, hits=hits)

    return all_events

def hitc_to_df_(hitc):
    columns = defaultdict(list)
    for hit in hitc.hits:
        columns["event"   ].append(hitc.event)
        columns["time"    ].append(hitc.time)
        columns["npeak"   ].append(hit .npeak)
        columns["Xpeak"   ].append(hit .Xpeak)
        columns["Ypeak"   ].append(hit .Ypeak)
        columns["nsipm"   ].append(hit .nsipm)
        columns["X"       ].append(hit .X)
        columns["Y"       ].append(hit .Y)
        columns["Xrms"    ].append(hit .Xrms)
        columns["Yrms"    ].append(hit .Yrms)
        columns["Z"       ].append(hit .Z)
        columns["Q"       ].append(hit .Q)
        columns["E"       ].append(hit .E)
        columns["Qc"      ].append(hit .Qc)
        columns["Ec"      ].append(hit .Ec)
        columns["track_id"].append(hit .track_id)
        columns["Ep"      ].append(hit .Ep)
    return pd.DataFrame(columns)






def soph_to_lowTh(df, threshold = 5):
    '''
    Converts sophronia 'RECO/Events' to lowTh events via a rather convoluted process
    Made by me (John Waiton), so dont treat it like a normal function from IC!
    ------
    Parameters
    ------
    df : pd.DataFrame
        DataFrame with obligatory columns :
                event, npeak, X, Y, Z,  Q, E
    threshold: int
        value at which the threshold is set.
    ------
    Returns
    ------
    Dictionary {event_number : HitCollection}
    from here
    '''

    # safety check, to ensure you don't accidentally make a repeating dataframe
    


    # new parameters for threshold, this is silly but I'm copying previous convention
    pes = 1
    threshold = threshold * pes
    same_peak = True

    # convert sophronia RECO/Events to hit collection
    print("Collecting hits from dataframe...")
    soph_hitc = hits_from_df(df)

    # collect the keys as the event numbers
    soph_hitc_list = list(soph_hitc.keys())

    print("Processing data...")
    # loop over all of these events
    j = 0
    for i in soph_hitc_list:
        j += 1

        if (len(soph_hitc_list)%j == 50): 
            print("{}/{}".format(j, len(soph_hitc_list)))
        # choose i'th event
        soph_hit_event = soph_hitc.get(i)

        # Apply threshold calculation
        soph_hitc_lowTh = threshold_hits(threshold, same_peak, soph_hit_event)

        # convert back to pandas dataframe using hitc_to_df
        soph_hdst_lowTh = hitc_to_df_(soph_hitc_lowTh)

        # check if pandas dataframe with all the events exists yet
        if 'full_soph_df' in locals() and isinstance(full_soph_df, pd.DataFrame):
            full_soph_df = pd.concat([full_soph_df, soph_hdst_lowTh])
        else:
            full_soph_df = soph_hdst_lowTh.copy(deep = True)
    
    return full_soph_df

def collect_min_max_bins(hits):
    '''
    returns all the min, max and mid values you'd need
    as well as the bins
    '''
    x_range = (hits.X.max()-hits.X.min())/2.
    y_range = (hits.Y.max()-hits.Y.min())/2.
    z_range = (hits.Z.max()-hits.Z.min())/2.
    mid_x   = (hits.X.max()+hits.X.min())/2.
    mid_y   = (hits.Y.max()+hits.Y.min())/2.
    mid_z   = (hits.Z.max()+hits.Z.min())/2.
    min_x = hits.X.min()
    min_y = hits.Y.min()
    min_z = hits.Z.min()

    max_x = hits.X.max()
    max_y = hits.Y.max()
    max_z = hits.Z.max()
    #print("X maximum and minimum")
    #print(max_x, min_x)
    #print("")

    #print("Y maximum and minimum")
    #print(max_y, min_y)
    #print("")

    #print("Z maximum and minimum")
    #print(max_z, min_z)

    xbins = int(hits.X.max()-hits.X.min())
    ybins = int(hits.Y.max()-hits.Y.min())
    zbins = int((hits.Z.max()-hits.Z.min())/2.)
    
    array = [x_range, y_range, z_range, mid_x, mid_y, mid_z, min_x, min_y, min_z, max_x, max_y, max_z, xbins, ybins, zbins]
    
    return (array)



def count_tracks_mc(hits_deco):
   
    # stuff needed for paolina track counting
    energy_threshold = 10
    min_voxels = 3
    
    base_vsize = 12 #mm
    the_hits = []

    xs = hits_deco.x
    ys = hits_deco.y
    zs = hits_deco.z
    es = hits_deco.energy

    for x, y, z, e in zip(xs, ys, zs, es):
        if np.isnan(e): continue
        h = Hit(0, Cluster(0, xy(x,y), xy(0,0), 0), z, e*1000, xy(0,0))
        the_hits.append(h)

    voxels = voxelize_hits(the_hits,
                           np.array([base_vsize, base_vsize, base_vsize]), False)

    (mod_voxels, dropped_voxels) = drop_end_point_voxels(voxels, energy_threshold, min_voxels)

    
    tracks = make_track_graphs(mod_voxels)
    tracks = sorted(tracks, key=get_track_energy, reverse = True)
    
    track_no = 0
    for c, t in enumerate(tracks, 0):
        track_no += 1
    
    return track_no    

def return_id(number):
    return str(df_ps[df_ps.particle_id == number].particle_name.values).strip("'[]'").split("'")[0]


def count_tracks(hits_deco):
    
    
    
    # stuff needed for paolina track counting
    energy_threshold = 10
    min_voxels = 3
    
    base_vsize = 12 #mm
    the_hits = []

    xs = hits_deco.X
    ys = hits_deco.Y
    zs = hits_deco.Z
    es = hits_deco.E

    for x, y, z, e in zip(xs, ys, zs, es):
        if np.isnan(e): continue
        h = Hit(0, Cluster(0, xy(x,y), xy(0,0), 0), z, e*1000, xy(0,0))
        the_hits.append(h)

    voxels = voxelize_hits(the_hits,
                           np.array([base_vsize, base_vsize, base_vsize]), False)

    (mod_voxels, dropped_voxels) = drop_end_point_voxels(voxels, energy_threshold, min_voxels)

    
    tracks = make_track_graphs(mod_voxels)
    tracks = sorted(tracks, key=get_track_energy, reverse = True)
    
    track_no = 0
    for c, t in enumerate(tracks, 0):
        track_no += 1
    
    return track_no

##################################################################################################
######################### FUNCTIONS THAT ARE USED TO CREATE THE SIGNAL FILES #####################
##################################################################################################

# develop a function that can load in files, collect them like this, and add to a large dataframe.
def collect_signal_df(data_path, save_path = "", verbose = True):
    """
    Function that collects Tl208 signal events (identified with Xe ions created in 'conv' processes)
    """

     # collect all filenames
    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.h5')]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")

    # counter for creating first array
    i = 0


    for file in file_names:
        file_path = data_path + file

        # load in data
        MC_df = pd.read_hdf(file_path, 'MC/particles')
        
        # set first dataframe
        if (i == 0):
            MC_signal_df = pd.DataFrame(columns = MC_df.columns.values)
            i = 1
        
        # collect conv slice and iterate over, scanning for Xe*** events
        conv_slice = MC_df[MC_df.creator_proc == 'conv']
        for e_id, df in conv_slice.groupby('event_id'):
            # check if conv selection has Xe isotope
            lst = df.particle_name.to_list()
            # truncate to only take 'Xe' as the input
            short_lst = []
            [short_lst.append(i[0:2]) for i in lst]
            if "Xe" in short_lst:
                if (verbose == True):
                    display(df)
                    print("Xe found in event {}".format(e_id))
                MC_signal_df = pd.concat([MC_signal_df, MC_df[MC_df.event_id == e_id]])

    # save if desired
    if save_path != "":
        MC_signal_df.to_hdf(str(save_path) + 'Tl_signal.h5', key = 'MC')

    # return dataframe    
    return MC_signal_df

def collect_soph_signal_df(data_path, event_ids, save_path = "", MC_check = True):
    '''

    Function that collects all sophronia signal events and stores them in a file

    This will produce a dataframe with all the 'RECO/Events' data
    for the relevant events.

    data_path -> path to the sophronia files
    event_ids -> array including all the event ids
    save_path -> path under which to save the data
    MC_check  -> check if the event ids are from MC, if so, they are doubled
    '''

    if MC_check == True:
        event_ids = event_ids * 2
         # collect all filenames
    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.h5')]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")

    # counter for creating first array
    i = 0


    #

    for file in file_names:
        file_path = data_path + file

        # load in data
        soph_df = dstio.load_dst(file_path, 'RECO', 'Events')
        
        # set first dataframe
        if (i == 0):
            soph_signal_df = pd.DataFrame(columns = soph_df.columns.values)
            i = 1

        
        # collect unique event ids from file, and then find how many are signals 
        ids = np.array(soph_df.event.unique())
        available_evts = np.intersect1d(ids, event_ids)

        # collect the signal events from the files
        for i in available_evts:
            soph_signal_df = pd.concat([soph_signal_df, soph_df[soph_df.event == i]])
    
        # save if desired
    if save_path != "":
        soph_signal_df.to_hdf(str(save_path) + 'Tl_signal_soph.h5', key = 'MC')

    return soph_signal_df


def collect_signal_true_hits(data_path, event_ids, save_path = ""):
    '''

    Function that collects all sophronia signal events and stores them in a file

    This will produce a dataframe with all the 'RECO/Events' data
    for the relevant events.

    data_path -> path to the sophronia files
    event_ids -> array including all the event ids
    save_path -> path under which to save the data
    MC_check  -> check if the event ids are from MC, if so, they are doubled
    '''

    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.h5')]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")

    # counter for creating first array
    i = 0


    #

    for file in file_names:
        file_path = data_path + file

        # load in data
        true_info = mcio.load_mchits_df(file_path).reset_index()
        
        # set first dataframe
        if (i == 0):
            true_info_df = pd.DataFrame(columns = true_info.columns.values)
            i = 1

        
        # collect unique event ids from file, and then find how many are signals 
        ids = np.array(true_info.event_id.unique())
        available_evts = np.intersect1d(ids, event_ids)

        # collect the signal events from the files
        for r in available_evts:
            true_info_df = pd.concat([true_info_df, true_info[true_info.event_id == r]])
    
        # save if desired
    if save_path != "":
        true_info_df.to_hdf(str(save_path) + 'Tl_signal_true_info.h5', key = 'MC')

    return true_info_df


def collect_signal_ids(input_dir, output_dir = r'/gluster/data/next/notebooks/john_books/soph_df_data/', PORT_CHOICE = '1a'):
    '''
    input_dir   -> directory just before the 'PORT_XX' directories. Takes sophronia input
                   like so: input_dir/PORT_1a/sophronia/
    output_dir  -> directory that all the relevant files should be dropped in
    PORT_CHOICE -> port of interest
    '''

    # collect the signal_ids    
    data = collect_signal_df(input_dir + 'PORT_' + PORT_CHOICE + '/sophronia/', output_dir + PORT_CHOICE + '_', verbose = False)
    signal_ids = np.array(data.event_id.unique())

    # save signal ids
    np.save(output_dir + 'signal_ids_' + PORT_CHOICE + '.npy', signal_ids)
    q = np.load(output_dir + 'signal_ids_' + PORT_CHOICE + '.npy', allow_pickle = True)

    # collect true information
    soph_true_data = collect_signal_true_hits(input_dir + 'PORT_' + PORT_CHOICE + '/sophronia/', q, save_path = output_dir + PORT_CHOICE + '_')

    # collect the sophronia data
    soph_data = collect_soph_signal_df(input_dir + 'PORT_' + PORT_CHOICE + '/sophronia/', q, save_path = output_dir + PORT_CHOICE + '_', MC_check = True)


# set folder_path here!
if __name__ == '__main__':
    # make this the full path
    input_d = sys.argv[1]
    output_d = sys.argv[2]
    PORT_CHOICE = sys.argv[3]
    print("Processing data at:\n{}".format(input_d))
    collect_signal_ids(str(input_d), str(output_d), str(PORT_CHOICE))
                                           
