import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tables as tb
from matplotlib import colors 

from typing          import Optional
from typing          import Union
from typing          import Callable

import sys,os,os.path
sys.path.append("/gluster/data/next/software/IC_311024/")
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
#sys.path.append("/home/e78368jw/Documents/NEXT_CODE/next_misc/")
os.environ['ICTDIR']='/gluster/data/next/software/IC_311024/'


from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
rcParams['figure.figsize'] = [10, 8]
rcParams['font.size'] = 22

import pandas as pd
import numpy  as np
import tables as tb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as clrs

import IC.invisible_cities.core.core_functions                   as     coref
import IC.invisible_cities.io.dst_io                           as     dstio

from IC.invisible_cities.cities                 import beersheba as beerfun

from IC.invisible_cities.evm.event_model                          import HitCollection

from IC.invisible_cities.database.load_db       import DataSiPM

from IC.invisible_cities.evm.event_model        import Cluster, Hit
from IC.invisible_cities.types.ic_types         import xy
from IC.invisible_cities.reco.paolina_functions import voxelize_hits

from IC.invisible_cities.evm.event_model        import HitEnergy
from IC.invisible_cities.cities.beersheba          import DeconvolutionMode
from IC.invisible_cities.cities.beersheba          import CutType

from IC.invisible_cities.reco import hits_functions as hif

from IC.invisible_cities.reco.deconv_functions import deconvolve
from IC.invisible_cities.reco.deconv_functions import deconvolution_input
from IC.invisible_cities.reco.deconv_functions import InterpolationMethod

import IC.invisible_cities.io.mcinfo_io as mcio

from IC.invisible_cities.cities.components import track_blob_info_creator_extractor
from IC.invisible_cities.io.hits_io        import load_hits
from IC.invisible_cities.io.hits_io        import hits_from_df
from IC.invisible_cities.core              import system_of_units as units
from IC.invisible_cities.types.symbols     import HitEnergy
from IC.invisible_cities.evm.event_model   import HitCollection
from IC.invisible_cities.evm.event_model   import Cluster
from IC.invisible_cities.evm.event_model   import Hit
from IC.invisible_cities.types.ic_types    import xy
from IC.invisible_cities.reco.paolina_functions import voxelize_hits

from concurrent.futures import ProcessPoolExecutor

from collections import defaultdict

import matplotlib.cm as cm
from matplotlib.colors import Normalize

pd.set_option('display.max_rows', 200)

import imageio
from pathlib import Path

############################################################################################
############################################################################################
#####################       TOPOLOGICAL PLOTTING
############################################################################################
############################################################################################


def plot_voxels(df, base_vsize = 12):

    xs = df.X
    ys = df.Y
    zs = df.Z
    es = df.E

    the_hits = []
    for x, y, z, e in zip(xs, ys, zs, es):
        if np.isnan(e): continue
        h = Hit(0, Cluster(0, xy(x,y), xy(0,0), 0), z, e*1000, xy(0,0))
        the_hits.append(h)
    
    voxels = voxelize_hits(the_hits,
                           np.array([base_vsize, base_vsize, base_vsize]), False)
    
    vsizex = voxels[0].size[0]
    vsizey = voxels[0].size[1]
    vsizez = voxels[0].size[2]

    min_corner_x = min(v.X for v in voxels) - vsizex/2.
    min_corner_y = min(v.Y for v in voxels) - vsizey/2.
    min_corner_z = min(v.Z for v in voxels) - vsizez/2.

    
    x = [np.round(v.X/vsizex) for v in voxels]
    y = [np.round(v.Y/vsizey) for v in voxels]
    z = [np.round(v.Z/vsizez) for v in voxels]
    e = [v.E for v in voxels]

    x_min = int(min(x))
    y_min = int(min(y))
    z_min = int(min(z))

    x_max = int(max(x))
    y_max = int(max(y))
    z_max = int(max(z))

    VOXELS = np.zeros((x_max-x_min+1, y_max-y_min+1, z_max-z_min+1))
    #print(VOXELS.shape)

    # sort through the event set the "turn on" the hit voxels
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=max(e))

    colors = np.empty(VOXELS.shape, dtype=object)
    for q in range(0,len(z)):
        VOXELS[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = 1
        colors[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = cmap(norm(e[q]))

    # and plot everything
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')
    #a,b,c is spacing in mm needs an extra dim
    a,b,c = np.indices((x_max-x_min+2, y_max-y_min+2, z_max-z_min+2))
    a = a*vsizex + min_corner_x
    b = b*vsizey + min_corner_y
    c = c*vsizez + min_corner_z

    # a, b, c are the corners of the voxels
    ax.voxels(a,b,c, VOXELS, facecolors=colors , edgecolor='k',alpha=0.8)

    ax.set_xlabel('x (mm)')#,fontsize=16)
    ax.set_ylabel('y (mm)')#,fontsize=16)
    ax.set_zlabel('z (mm)')#,fontsize=16)


    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm)
    cb.set_label('Energy (keV)')

    fig.suptitle('voxelised')

    #ax.view_init(-160, 90)

    plt.show(fig)


def raw_plotter(q, evt, pitch = 15.55):
    '''
    just plots the hits, nothing smart
    '''

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    xx = np.arange(q.X.min(), q.X.max() + pitch, pitch)
    yy = np.arange(q.Y.min(), q.Y.max() + pitch, pitch)
    zz = np.sort(q.Z.unique())

    axes[0].hist2d(q.X, q.Y, bins=[xx, yy], weights=q.Q, cmin=0.0001);
    axes[0].set_xlabel('X (mm)');
    axes[0].set_ylabel('Y (mm)');

    axes[1].hist2d(q.X, q.Z, bins=[xx, zz], weights=q.Q, cmin=0.0001);
    axes[1].set_xlabel('X (mm)');
    axes[1].set_ylabel('Z (mm)');


    axes[2].hist2d(q.Y, q.Z, bins=[yy, zz], weights=q.Q, cmin=0.0001);
    axes[2].set_xlabel('Y (mm)');
    axes[2].set_ylabel('Z (mm)');
    fig.suptitle("rebinned in Z")
    plt.show(fig)




def plotter(df, evt, cut_n_drop = True, show = True, deconv = False, cut_sensors = None, drop_sensors = None, c_min = 0.0001):
    evt_interest = df[df.event == evt]
    
    if deconv == False:
        print('======================')
        print(f'EVENT {evt}')
        E_evt = evt_interest.Ec.sum()
        print(f'Total energy {E_evt:.2f} MeV')
        print('======================\n')
        weight = 'Q'
    else:
        print('======================')
        print(f'EVENT {evt}')
        E_evt = evt_interest.E.sum()
        print(f'Total energy {E_evt:.2f} MeV')
        print('======================\n')
        weight = 'E'        
    if cut_n_drop == True:
        hits_cut = coref.timefunc(cut_sensors)(evt_interest.copy())
        hits_drop = coref.timefunc(drop_sensors)(hits_cut.copy())
    else:
        hits_drop = evt_interest
    pitch = 15.55
    # then applying transformations to convert to 'SiPM outputs'
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    xx = np.arange(hits_drop.X.min(), hits_drop.X.max() + pitch, pitch)
    yy = np.arange(hits_drop.Y.min(), hits_drop.Y.max() + pitch, pitch)
    zz = np.sort(hits_drop.Z.unique())
    # Pad zz bins to ensure no bin stretches larger than 4
    zz_diff = np.diff(zz)
    zz_padded = []
    for i, z in enumerate(zz[:-1]):
        zz_padded.append(z)
        if zz_diff[i] > 4:
            zz_padded.extend(np.arange(z + 4, zz[i + 1], 4))
    zz_padded.append(zz[-1])
    zz = np.array(zz_padded)

    axes[0].hist2d(hits_drop.X, hits_drop.Y, bins=[xx, yy], weights=hits_drop[weight], cmin=c_min);
    axes[0].set_xlabel('X (mm)');
    axes[0].set_ylabel('Y (mm)');

    axes[1].hist2d(hits_drop.X, hits_drop.Z, bins=[xx, zz], weights=hits_drop[weight], cmin=c_min);
    axes[1].set_xlabel('X (mm)');
    axes[1].set_ylabel('Z (mm)');

    axes[2].hist2d(hits_drop.Y, hits_drop.Z, bins=[yy, zz], weights=hits_drop[weight], cmin=c_min);
    axes[2].set_xlabel('Y (mm)');
    axes[2].set_ylabel('Z (mm)');

    fig.suptitle(f'Processed Sensors Signal_{evt} - E: {E_evt}', fontsize=30)
    #plt.savefig(f'plots/hits_{evt}.pdf')
    if show == True:
        plt.show(fig)



def plotter_3d(df, evt, cut_n_drop = True, show = True, clrbar = True, alpha = 0.65, min_s = 1, max_s = 15, cut_sensors = None, drop_sensors = None):
    '''
    evt_interest - df
    evt          - event number

    '''
    # plot
    evt_interest = df[df.event == evt]

    if cut_n_drop:
        hits_cut = coref.timefunc(cut_sensors)(evt_interest.copy())
        hits_drop = coref.timefunc(drop_sensors)(hits_cut.copy())
    else:
        hits_drop = evt_interest.copy(deep = True)
    
    xt = hits_drop.X
    yt = hits_drop.Y
    zt = hits_drop.Z
    et = hits_drop.E
    
    fig = plt.figure()
    #fig.suptitle('3D post deconvolution ' + str(evt), fontsize=30)
    fig.suptitle(f'Electron candidate event {evt}', fontsize=30)
    ax = fig.add_subplot(111, projection='3d')
    
    
    
    ets = et > 0 # eliminate small things for measurement
    
    max_val = max(et[ets])
    scaled_clipped = [max((v / max_val) * max_s, min_s) for v in et[ets]]

    #p = ax.scatter(x[em], y[em], z[em], c=e[em], alpha=0.3, cmap='viridis')
    #plt_sphere([(-track.blob2_x.values[0], -track.blob2_y.values[0], -track.blob2_z.values[0])], [blobR])
    p = ax.scatter([xt[ets]], yt[ets], zt[ets], c=et[ets], alpha=alpha, cmap='viridis', s = scaled_clipped)#, s = et[ets])
    #q = ax.scatter(xt, yt, zt, alpha = 0.3, color = 'red')
    
    # overlay the blobs and their radii
    if clrbar:
        cb = fig.colorbar(p, ax=ax)
        cb.set_label('Energy (keV)')
    
    
    ax.set_xlabel('\nx (mm)')
    ax.set_ylabel('\ny (mm)')
    ax.set_zlabel('\nz (mm)')
    
    #ax.set_xlim([-300, -100])
    #ax.set_ylim([250, 450])
    #ax.set_zlim([1600, 1800])
    #ax.view_init(20, -150)
    
    #plt.savefig(f'gif_making/deconv/angle_{i}.png')
    #plt.savefig(f'plots/hits_3d_{evt}.pdf')
    if show:
        plt.show()

    return hits_drop


def plotter_blobs(df_beer, df_isau, evt, show = False):
    '''
    Plot beersheba tracks with blobs overlaid
    '''

    evt_beer = df_beer[df_beer.event == evt]
    evt_isau = df_isau[df_isau.event == evt]

    pitch = 15.55
    # then applying transformations to convert to 'SiPM outputs'
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))


    xx = np.arange(evt_beer.X.min(), evt_beer.X.max() + pitch, pitch)
    yy = np.arange(evt_beer.Y.min(), evt_beer.Y.max() + pitch, pitch)
    zz = evt_beer.Z.unique()

    axes[0].hist2d(evt_beer.X, evt_beer.Y, bins=[xx, yy], weights=evt_beer['E'], cmin=0.0001);
    axes[0].set_xlabel('X (mm)');
    axes[0].set_ylabel('Y (mm)');

    axes[1].hist2d(evt_beer.X, evt_beer.Z, bins=[xx, zz], weights=evt_beer['E'], cmin=0.0001);
    axes[1].set_xlabel('X (mm)');
    axes[1].set_ylabel('Z (mm)');


    axes[2].hist2d(evt_beer.Y, evt_beer.Z, bins=[yy, zz], weights=evt_beer['E'], cmin=0.0001);
    axes[2].set_xlabel('Y (mm)');
    axes[2].set_ylabel('Z (mm)');

############################################################################################
############################################################################################
#####################       OTHER PLOTTING
############################################################################################
############################################################################################

def plot_hist(df, column = 'energy', binning = 20, title = "Energy plot", output = False, fill = True, label = 'default', x_label = 'energy (MeV)', y_label = 'Counts/bin', range = 0, outliers = None, log = True, data = False, save = False, save_dir = '', alpha = 1):
    '''
    Produce a histogram for a specific column within a dataframe.
    Used for NEXT-100 data analysis. Can be overlaid on top of other histograms.
    
    Args:
        df          :       pandas dataframe
        column      :       column the histogram will plot
        binning     :       number of bins in histogram
        title       :       title of the plot
        output      :       visualise the plot (useful for notebooks)
        fill        :       fill the histogram
        label       :       Add histogram label (for legend)
        x_label     :       x-axis label
        y_label     :       y-axis label
        range       :       range limiter for histogram (min, max)
        outliers    :       Remove outliers by percentile (min, max)
                            If you set a range, this wont work
        log         :       y-axis log boolean
        data        :       output histogram information boolean
        save        :       Save the plot as .png boolean
        save_dir    :       directory to save the plot
        alpha       :       opacity of histogram

    Returns:
        if (data==False):
            None          :       empty return
        if (data==True):
            (cnts,        :       number of counts
            edges,        :       values of the histogram edges
            patches)      :       matplotlib patch object


    '''
    # for simplicity/readability, scoop out the relevant columns from the pandas dataframe.
    energy_vals = df[column].to_numpy()

    
    if outliers is not None:
        lower = np.percentile(energy_vals, outliers[0])
        upper = np.percentile(energy_vals, outliers[1])
        energy_vals  = energy_vals[(energy_vals >= lower) & (energy_vals <= upper)]

    if (range==0):
        range = (np.min(energy_vals), np.max(energy_vals))
    
    # control viewing of hist
    if (fill == True):
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, range = range, alpha = alpha, color='C1')
    else:
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, histtype='step', linewidth = 2, range = range)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if log == True:
        plt.yscale('log')
    if (save==True):
        if not (save_dir == ''):
            plt.savefig(save_dir + title + ".png")
        else:
            print("Please provide a suitable directory to save the data!")
    if (output==True):
        plt.legend()
        plt.show()
    if (data==True):
        return (cnts, edges, patches)
    else:
        return
    

def plot_hist_over_column(df, grouped_column = 'event', column = 'energy', function = lambda x: x.sum(), binning = 20, title = "Energy plot", output = False, fill = True, label = 'default', x_label = 'energy (MeV)', range = 0, log = True, data = False, save = False, save_dir = '', alpha = 1):
    '''
    Produce a histogram for a specific column within a dataframe grouped by another
    and functionalised
    Used for NEXT-100 data analysis. Can be overlaid on top of other histograms.
    
    Args:
        df              :       pandas dataframe
        grouped_column  :       column to be grouped by
        column          :       column the histogram will plot
        function        :       function to apply to the grouped column (sum, mean, etc)
                                must be passed in as a lambda function
        binning         :       number of bins in histogram
        title           :       title of the plot
        output          :       visualise the plot (useful for notebooks)
        fill            :       fill the histogram
        label           :       Add histogram label (for legend)
        x_label         :       x-axis label
        range           :       range limiter for histogram (min, max)
        log             :       y-axis log boolean
        data            :       output histogram information boolean
        save            :       Save the plot as .png boolean
        save_dir        :       directory to save the plot
        alpha           :       opacity of histogram

    Returns:
        if (data==False):
            None          :       empty return
        if (data==True):
            (cnts,        :       number of counts
            edges,        :       values of the histogram edges
            patches)      :       matplotlib patch object


    '''    

    vals = function(df.groupby(grouped_column)[column])

    if (range==0):
        range = (np.min(vals), np.max(vals))

    # control viewing of hist
    if (fill == True):
        cnts, edges, patches = plt.hist(vals, bins = binning, label = label, range = range, alpha = alpha)
    else:
        cnts, edges, patches = plt.hist(vals, bins = binning, label = label, histtype='step', linewidth = 2, range = range)
    plt.title(title)
    plt.ylabel("events")
    plt.xlabel(x_label)
    if log == True:
        plt.yscale('log')
    if (save==True):
        if not (save_dir == ''):
            plt.savefig(save_dir + title + ".png")
        else:
            print("Please provide a suitable directory to save the data!")
    if (output==True):
        plt.legend()
        plt.show()
    if (data==True):
        return (cnts, edges, patches)
    else:
        return

############################################################################################
############################################################################################
#####################               CUTS
############################################################################################
############################################################################################



def cut_effic(df1, df2, verbose = False):
    '''
    Produce efficiency of a single cut by comparison of unique
    values in two databases (pre and post cut).
    This works best with h5 files produced through IC chain.

    Args:
        df1         :       dataframe 1 (post-cut)
        df2         :       dataframe 2 (pre-cut)
        verbose     :       verbose boolean

    Returns:
        efficiency  :       efficiency of cuts
    '''

    length_1 = df1['event'].nunique()
    length_2 = df2['event'].nunique()
    efficiency = ((length_1/length_2)*100)
    print("Efficiency: {:.2f} %".format(efficiency))

    if (verbose == True):
        print(("Events in reduced dataframe: {}\nEvents in initial dataframe: {}").format(len(df1), len(df2)))

    return efficiency


def fiducial_track_cut_2(df, lower_z = 20, upper_z = 1195, r_lim = 472, verbose = False):
    '''
    Remove events outwith the defined fiducial volume.

    Args:
        df          :           pandas dataframe
        lower_z     :           lower z-bound for fiducial cut
        upper_z     :           upper z-bound for fiducial cut
        r_lim       :           radial bound for fiducial cut
        verbose     :           verbose boolean

    Returns:
        df3         :           cut dataframe
    '''
    # create lists of outer_fiduc entries
    z_df_low = df[(df['z_min'] <= lower_z)]
    z_df_up = df[(df['z_max'] >= upper_z)]
    r_df = df[(df['r_max'] >= r_lim)]

    # scrape the events
    low_list = (z_df_low['event'].to_numpy())
    up_list = (z_df_up['event'].to_numpy())
    r_list = (r_df['event'].to_numpy())

    # apply the filter to remove all events that fall in outer fiduc
    df1 = df[~df['event'].isin(low_list)]
    df2 = df1[~df1['event'].isin(up_list)]
    df3 = df2[~df2['event'].isin(r_list)]

    if (verbose == True):
        print("Cutting events around fiducial volume related to:\nZ range between {} and {}\nRadius range < {}".format(lower_z, upper_z, r_lim))


    return df3




def one_track_cuts(df, verbose = False):
    '''
    Remove events with more than one track from dataframe
    There is a better way of doing this, using a column within the dataframe.

    Args:
        df                  :               pandas dataframe
        verbose             :               verbose boolean

    Returns:
        one_track_events    :               cut dataframe

    '''
    # 1-track event counter
    event_counts = df.groupby('event').size()
    #print(event_counts[:5]) # showing that you see how many 
                            #  trackIDs there are per event
    one_track = event_counts[event_counts == 1].index

    # filter dataframe
    one_track_events = df[df['event'].isin(one_track)]
    

    if (verbose == True):
        print("Removing events with more than one track.")
        print("Events with one track: {}".format(one_track))
        display(one_track_events.head())
    

    return one_track_events




def overlapping_cuts(df, verbose = False):
    '''
    Remove all events with blobs of overlapping energy != 0
    This has been written to work regardless of one-track cut being applied.

    Args:
        df              :               pandas dataframe
        verbose         :               verbose boolean

    Returns:
        ovlp_remove     :               cut dataframe
    '''

    ovlp_remove = df[df['ovlp_blob_energy']!=0]
    ovlp_remove = df[~df['event'].isin(ovlp_remove.event.unique())]

    if (verbose==True):
        print("Removing overlapping blobs....")

    return ovlp_remove




def energy_cuts(df, lower_e = 1.5, upper_e = 1.7, verbose = False):
    '''
    Remove all events outwith the relevant energy values
    This HAS NOT been written to work regardless of one-track cut being applied.

    Args:
        df              :           pandas dataframe
        lower_e         :           lower bound for energy
        upper_e         :           upper bound for energy
        verbose         :           verbose boolean
    
    Returns:
        filt_e_df       :           cut dataframe
    '''
    filt_e_df = df[(df['energy'] >= lower_e) & (df['energy'] <= upper_e)]

    if (verbose == True):
        print("Cutting energy events around {} & {} keV".format(lower_e, upper_e))

    return filt_e_df


def remove_low_E_events(df, energy_limit = 0.05):
    '''
    Remove satellite energy tracks, add their energy back to the
    first track and then update 'numb_of_tracks' to be accurate

    Args:
        df              :           pandas dataframe
        energy_limt     :           upper bound of energy

    Returns:
        remove_low_E    :           cut dataframe
    '''

    tracks_test = df.copy(deep=True)

    # take events with lower than 50 keV, 0.05 MeV
    condition = (tracks_test.energy < energy_limit)
    summed_df = tracks_test[condition].groupby('event')['energy'].sum().reset_index()

     # merge these as a new column
    merged_df = pd.merge(tracks_test, summed_df, on='event', suffixes=('', '_sum'), how = 'left').fillna(0)
    # add this summed energy to first column
    merged_df['energy'] = merged_df.apply(lambda row: (row['energy'] + row['energy_sum']) if row.name == merged_df[merged_df['event'] == row['event']].index[0] else row['energy'], axis=1)

    # drop energy sum column
    result_df = merged_df.drop('energy_sum', axis = 1)

    # then remove all tracks below the energy threshold
    condition_upper = (result_df.energy > energy_limit)
    remove_low_E = result_df[condition_upper]

    # count the number of events identified with unique event, and change numb_of_tracks to reflect this
    event_counts = remove_low_E['event'].value_counts(sort = False)

    # apply this to numb_of_tracks
    remove_low_E['numb_of_tracks'] = remove_low_E['event'].map(event_counts)

    return remove_low_E


def len_events(df, tag = 'event'):
    '''
    Returns the number of unique events as len(df) doesn't work in
    our case (IC chain)

    Args:
        df          :       pandas dataframe
        tag         :       tag to know which column to check for unique events
    
    Returns:
        length_1    :       length of dataframe
    '''
    length_1 = df[tag].nunique()
    return length_1


def apply_cuts(tracks, lower_z = 20, upper_z = 1195, r_lim = 472, lower_e = 1.5, upper_e = 1.7, overlap_flag = True):
    '''
    Applies all known cuts, returns dataframe and efficiency table.
    Highly bespoke function, use with care.

    NOTE: Does not include satellite track removal

    Args:
        tracks          :       dataframe of particle tracks
        lower_z         :       lower z-bound for fiducial cut
        upper_z         :       upper z-bound for fiducial cut
        r_lim           :       radial bound for fiducial cut
        lower_e         :       lower bound for energy cut
        upper_e         :       upper bound for energy cut
    
    Returns:
        (ecut_rel,      :       dataframe with output of track cuts
        efficiencies)           efficiency table
    '''
    # Efficiency calculation
    cut_names = []
    rel_cut_effics = []
    abs_cut_effics = []
    cut_events = []

    # no cuts
    cut_names.append("No cuts")
    rel_cut_effics.append(100)
    abs_cut_effics.append(100)
    # number of events
    cut_events.append(len_events(tracks))


    #####################################################################
    #####################################################################

    # fiducial cuts
    cut_names.append("Fiducial Cuts")

    # make fiducial cuts
    fiducial_rel = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = False)

    fiducial_abs = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = True)

    # make efficiency calculation
    print("Fiducial track cut")
    print("==================")
    print("Relative Cut efficiency:")
    ef = cut_effic(fiducial_rel, tracks)
    rel_cut_effics.append(ef)
    cut_events.append(len_events(fiducial_rel))

    print('Absolute Cut efficiency:')
    ef = cut_effic(fiducial_abs, tracks)
    abs_cut_effics.append(ef)



    #####################################################################
    #####################################################################

    cut_names.append("One track cut")
    one_track_rel = one_track_cuts(fiducial_rel, verbose = False)

    # events are relative, as absolute efficiency lets you figure out events from the beginning# absolute
    one_track_abs = one_track_cuts(tracks)

    # relative
    print("One track cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(one_track_rel, fiducial_rel)
    rel_cut_effics.append(ef)
    cut_events.append(len_events(one_track_rel))

    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(one_track_abs, tracks)
    abs_cut_effics.append(ef)



    #####################################################################
    #####################################################################

    if overlap_flag:
        # apply cuts
        ovlp_rel = overlapping_cuts(one_track_rel)
        ovlp_abs = overlapping_cuts(tracks)
    else:
        ovlp_rel = one_track_rel
        ovlp_abs = tracks


    cut_names.append("Blob overlap cuts")

    # relative
    print("\nBlob overlap cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(ovlp_rel, one_track_rel)
    rel_cut_effics.append(ef)

    cut_events.append(len_events(ovlp_rel))
    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(ovlp_abs, tracks)
    abs_cut_effics.append(ef)


    #####################################################################
    #####################################################################

    ecut_rel = energy_cuts(ovlp_rel, lower_e, upper_e)
    ecut_abs = energy_cuts(tracks, lower_e, upper_e)

    cut_names.append("Energy cuts")

    # relative
    print("Energy cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(ecut_rel, ovlp_rel)
    rel_cut_effics.append(ef)
    cut_events.append(len_events(ecut_rel))


    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(ecut_abs, tracks)
    abs_cut_effics.append(ef)

    efficiencies = pd.DataFrame({'Cut': cut_names,
                             'Relative Efficiency': rel_cut_effics,
                             'Relative Events': cut_events,
                             'Single Cut Efficiency': abs_cut_effics
                             })


    # adding exception in for when there's no data in ecut_rel
    if (len(ecut_rel.index) == 0):
            #efficiencies.loc[len(efficiencies.index)] = ['pos_evt - all_evt', 0, len(ecut_rel), 0]
            #efficiencies.loc[len(efficiencies.index)] = ['FOM_MAX - blob2_E_val (MeV)', 0, 0, 0]
            #efficiencies.to_csv(str(folder_path) + 'output/efficiency.csv')
            print("No events left in ROI... jobs done!")
            #return 0
    return (ecut_rel, efficiencies)




############################################################################################
############################################################################################
#####################               LOADING IN OF DATA
############################################################################################
############################################################################################



def load_single_file(file_path, group, node):
    '''
    Load data from a single h5 file and produce dataframes for /group/node

    Args:
        file_path       :       str
                                Path to the h5 file to be loaded.

    Returns:
        tracks_df       :       pandas.DataFrame
                                DataFrame containing the /group/node data.
    '''
    try: 
        tracks_df = load_dst(file_path, group, node)
        return tracks_df
    except Exception as e:
        print(f'File {file_path} broke with error:\n{e}')
        x = pd.DataFrame()
        return x


def load_data_fast(folder_path, group, node):
    '''
    Load multiple h5 files and produce concatenated dataframes for /Tracking/Tracks, /MC/Particles, and their corresponding eventmap.

    Args:
        folder_path     :       str
                                Path to the folder containing the h5 files.

    Returns:
        tracks          :       pandas.DataFrame
                                Concatenated DataFrame containing the /Tracking/Tracks data from all h5 files.
        
        particles       :       pandas.DataFrame
                                Concatenated DataFrame containing the /MC/particles data from all h5 files, with the 'event_id' column modified.

        eventmap        :       pandas.DataFrame
                                Concatenated DataFrame containing the event map from all h5 files.
    '''
    
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    file_paths = [os.path.join(folder_path, f) for f in file_names]

    # Use ProcessPoolExecutor to parallelize the data loading process
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(lambda file_path: load_single_file(file_path, group, node), file_paths))
    
    # Separate the results into respective lists
    tracks_list = results

    # Concatenate all the dataframes at once
    tracks = pd.concat(tracks_list, axis=0, ignore_index=True)

    return tracks