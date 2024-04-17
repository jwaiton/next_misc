'''

AN OVERARCHING PYTHON SCRIPT THAT IM GOING TO FORCE ALL MY FUNCTIONS INTO ONE WAY OR ANOTHER
IM NOT COPYING AND PASTING THE 'LOAD_DATA' FUNCTION EVER AGAIN. I REFUSE.

Any time I need to use a function, I'm adding it to this stupidly long list of functions
'''



import sys,os,os.path
sys.path.append("../../") # if you move files around, you need to adjust this!
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/home/e78368jw/Documents/NEXT_CODE/IC'

import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import tables as tb
import IC.invisible_cities.io.dst_io                           as     dstio
import IC.invisible_cities.io.mcinfo_io as mcio
from    IC.invisible_cities.core.core_functions   import shift_to_bin_centers

import scipy.special as special
from scipy.stats import skewnorm
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker



def plot_hist(df, column = 'energy', binning = 20, title = "Energy plot", output = False, fill = True, label = 'default', x_label = 'energy (MeV)', range = 0, log = True, data = False, save = False, save_dir = '', alpha = 1):
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
        range       :       range limiter for histogram (min, max)
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

    if (range==0):
        range = (np.min(energy_vals), np.max(energy_vals))

    # control viewing of hist
    if (fill == True):
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, range = range, alpha = alpha)
    else:
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, histtype='step', linewidth = 2, range = range)
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
        plt.show()
    if (data==True):
        return (cnts, edges, patches)
    else:
        return


def load_data(folder_path):
    '''
    Load in multiple h5 files and produce dataframes corresponding to /Tracking/Tracks, /MC/Particles, and their corresponding
    eventmap.

    Args:
        folder_path     :       path to folder of h5 files
    Returns:
        (tracks,        :       tracks dataframe
        particles,      :       MC particle information dataframe
        eventmap)       :       eventmap for MC -> Tracks
    '''
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.h5')]
    
    # remove any files that dont end in h5

    # NOTE Break this section up, its annoying like this.
    dfs = []
    df_trs = []
    df_ems = []
    i = 0
    end = len(file_names)
    # create massive dataframe with all of them
    for file in file_names:
        file_path = folder_path + file
        df = dstio.load_dst(file_path, 'Tracking', 'Tracks')
        dfs.append(df)
        # include MC particles (boooo takes ages)

        # collecting the correct components of the file, not exactly sure how this works
        df_ps = pd.read_hdf(file_path, 'MC/particles')
        #df_ps = df_ps[df_ps.creator_proc == 'conv']
        # collecting event map
        df_em = mcio.load_eventnumbermap(file_path).set_index('nexus_evt')
        df_trs.append(df_ps)
        df_ems.append(df_em)
        i += 1

        if (i%50 == 0):
            print(i)

    tracks = pd.concat(dfs, axis=0, ignore_index=True)

    particles = pd.concat(df_trs, ignore_index=True)
    particles['event_id'] = particles['event_id'] * 2   # double it

    eventmap = pd.concat([dt for dt in df_ems])
    # create particle list also

    return (tracks, particles, eventmap)

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

    Args:
        df              :               pandas dataframe
        verbose         :               verbose boolean

    Returns:
        ovlp_remove     :               cut dataframe
    '''

    ovlp_remove = df[df['ovlp_blob_energy']==0]

    if (verbose==True):
        print("Removing overlapping blobs...")

    return ovlp_remove




def energy_cuts(df, lower_e = 1.5, upper_e = 1.7, verbose = False):
    '''
    Remove all events outwith the relevant energy values

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


def len_events(df):
    '''
    Returns the number of unique events as len(df) doesn't work in
    our case (IC chain)

    Args:
        df          :       pandas dataframe
    
    Returns:
        length_1    :       length of dataframe
    '''
    length_1 = df['event'].nunique()
    return length_1


def positron_scraper(data_path, save = False):
    """
    Function that iterates over files with MC and collects only positron events.
    Intended to reduce the memory resources of MC data.

    Args:
        data_path       :       path to folder of h5 files
        save            :       save the data separately boolean
    
    Returns:
        pos_df          :       positron dataframe
    """



     # collect all filenames
    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.h5')]
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
            #print("Chunking at event {}!".format(i))
            # concat the list
            MC_df = pd.concat(MC_df, axis = 0, ignore_index = True)
            #print("Post concat")
            #display(MC_df)
            pos_data = MC_df[MC_df['particle_name'] == 'e+']

            
            #display(pos_data)
            #print(type(pos_data))
            # collect positron events into df
            pos_df = pos_df.append(pos_data)
            #print("{} positron events found\n{} positron events total".format(pos_data.shape[0],pos_df.shape[0]))
            #display(pos_df)

            # make space
            MC_df = []

    if (save == True):
        pos_df.to_hdf('positrons.h5', key = 'pos', mode = 'w')

    return pos_df


def blob_positron_plot(ecut_rel, ecut_no_positron_df, save = False, save_title = 'plot.png'):
    '''
    Plots the blob energies with and without positrons.
    This is a very bespoke function.

    Args:
        ecut_rel                :          dataframe with positrons
        ecut_no_positron_df     :          dataframe with no positrons
        save                    :          save figure boolean
        save_title              :          figure title
    
    Returns:
        None                    :          empty return
    '''
    # make range full range of blob1 and blob2
    eblob_full = []
    eblob_full.append(ecut_rel['eblob1'].to_numpy())
    eblob_full.append(ecut_rel['eblob2'].to_numpy())
    minimum_e = np.min(eblob_full)
    maximum_e = np.max(eblob_full)

    # the original way
    plot_hist(ecut_rel, column = 'eblob2', binning = 20, title = "Blob energies", output = False, fill = False, label = 'blob 2', x_label = 'energy (MeV)', range = (minimum_e, maximum_e))
    plot_hist(ecut_rel, column = 'eblob1', binning = 20, title = "Blob energies", output = False, fill = False, label = 'blob 1', x_label = 'energy (MeV)', range = (minimum_e, maximum_e))

    #plt.hist(no_pos_blob1, bins = 20, label = 'events with no e+', range = (minimum_e, maximum_e))
    #plt.hist(no_pos_blob2, bins = 20, label = 'events with no e+', range = (minimum_e, maximum_e))

    plot_hist(ecut_no_positron_df, column = 'eblob1', binning = 20, title = "Blob energies", output = False, fill = True, label = 'blob1 - no e+', x_label = 'energy (MeV)', range = (minimum_e, maximum_e))
    plot_hist(ecut_no_positron_df, column = 'eblob2', binning = 20, title = "Blob energies", output = False, fill = True, label = 'blob2 - no e+', x_label = 'energy (MeV)', range = (minimum_e, maximum_e), alpha = 0.5)

    plt.legend()

    if (save == True):
        plt.savefig(save_title)
    plt.show()


def true_fom_calc(p_data, no_p_data, cut_list, verbose = False):
    '''
    Produces a figure of merit list based on cuts to 
    specific categories and their consequent fits.

    Args:
        p_data          :       positron event dataframe (signal)
        no_p_data       :       no positron event dataframe (background)
        cut_list        :       list of blob-2 energy cuts to apply across FOM scan
        verbose         :       verbose boolean

    Returns:
        (fom,           :       Array of FOM values across the cuts
        fom_err,        :       Array of FOM errors across the cuts
        ns,             :       Array of number of signal values across the cuts
        nb)             :       Array of number of background values across the cuts
    '''

    # create deep copies for safety
    pos_data = p_data.copy(deep = True)
    no_pos_data = no_p_data.copy(deep = True)

    # Take the initial, no blob2 cut values for ns and nb
    ns0 = len(pos_data.index)
    nb0 = len(no_pos_data.index)
    total0 = ns0 + nb0


    if (verbose == True):
        print("Total events: {}\nSignal events: {}\nBackground events: {}\n".format(total0, ns0, nb0))
        blob_positron_plot(pos_data, no_pos_data)
    
    

    # create all the lists for fom
    e = []
    e_err = []
    b = []
    b_err = []
    fom = []
    fom_err = []
    ns = [ns0]
    nb = [nb0]

    for i in range(len(cut_list)):
        
        # remove blob 2 values below value on cut_list
        pos_data = pos_data[(pos_data['eblob2'] > cut_list[i])]
        no_pos_data = no_pos_data[(no_pos_data['eblob2'] > cut_list[i])]


        # apply fit to the new data 
        if (verbose == True):
            print("Signal events: {}\nBackground events: {}\n FOM: {}".format())
        
        # collect number of signal events vs number of backgrounds, which you know 
        ns.append(len(pos_data.index))
        nb.append(len(no_pos_data.index))



        # produce fom value, if ns0 or nb is zero, set to zero.
        try:
            e.append(ns[i+1]/ns0)
        except:
            print("Zero-div error, appending 0")
            e.append(0)
        
        try:
            b.append(nb[i+1]/nb0)
        except ZeroDivisionError:
            print("Zero-div error, appending 0")
            b.append(0)
        fom.append(e[i]/np.sqrt(b[i]))

        # errors

        # errors for e and b


        # errors for fom
        e_err.append(ratio_error(e[i],ns[i+1],ns0,np.sqrt(ns[i+1]),np.sqrt(ns0)))
        b_err.append(ratio_error(b[i],nb[i+1],nb0,np.sqrt(nb[i+1]),np.sqrt(nb0)))
        fom_err.append(fom_error(e[i], b[i], e_err[i], b_err[i]))

        if (verbose == True):
            blob_positron_plot(pos_data, no_pos_data)
    
    # that should be it? i dont expect this to work first time, but lets test it!
    return (fom, fom_err, ns, nb)


def apply_cuts(tracks, lower_z = 20, upper_z = 1195, r_lim = 472, lower_e = 1.5, upper_e = 1.7):
    '''
    Applies all known cuts, returns dataframe and efficiency table.
    Highly bespoke function, use with care.

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

    # apply cuts
    ovlp_rel = overlapping_cuts(one_track_rel)
    ovlp_abs = overlapping_cuts(tracks)


    cut_names.append("Blob overlap cuts")

    # relative
    print("Blob overlap cut")
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

def apply_FOM(path, data, cut_list, plot = False, plot_title = " "):
    '''
    Function that applies the figure of merit calculation

    Args:
        path                :           data path for collecting positron events
        data                :           dataframe for tracks
        cut_list            :           list of cuts
        plot                :           plotting boolean
        plot_title          :           title of plot
    
    Returns:
        (positron_events,   :           Number of positron (signal) events
        len(data),                      Number of total events
        fom_max,                        Maximal FOM value
        blob_val)                       blob-2 cut value at maximal FOM value
        
    '''
    # collect positron events
    positron_events = positron_scraper(path)
    pos_events = (np.unique(positron_events['event_id'].to_numpy()))*2

    # number of events that are positrons
    ecut_positron_df = data[data['event'].isin(pos_events)]
    ecut_no_positron_df = data[~data['event'].isin(pos_events)]
    fom = true_fom_calc(ecut_positron_df, ecut_no_positron_df, cut_list)
    # sanitise
    ns = fom[2]
    nb = fom[3]
    fom_error = np.nan_to_num(fom[1])
    fom = np.nan_to_num(fom[0])

    print("ns, nb")
    print(ns)
    print(nb)
    

    print("FOM values:")
    print(fom)
    print("Errors")
    print(fom_error)

    # remove stupid values based on low statistics
    fom[fom > 10] = 0
    fom[fom < 0] = 0

    max_index = np.argmax(fom)
    # prep output for efficiencies
    positron_events = len(ecut_positron_df)
    fom_max = fom[max_index]
    blob_val = cut_list[max_index]

    if (plot == True):
        plt.errorbar(cut_list, fom, yerr = fom_error)
        plt.title(plot_title)
        plt.xlabel("Blob-2 energy threshold (MeV)")
        plt.legend()
        
        plt.ylabel("fom")
        plt.show()

        # this isn't correct (removing the last element), but i think it'll be okay
        plt.plot(cut_list, ns[:-1], label = "Signal events")
        plt.plot(cut_list, nb[:-1], label = "Background events")
        plt.title("signal and background across cuts")
        plt.legend()
        plt.show()

    return (positron_events, len(data), fom_max, blob_val)


######################################################################

# FUNCTIONS FROM THE FOM_PLOTTER

######################################################################

def plot_2Dhist(ND_array, xlabel, ylabel, title = '2D Histogram', xlabel_title = 'x axis', ylabel_title = 'y axis'):
    '''
    Plots 2D histogram from array of NxN dimensions. Note: must be a hstack array via numpy
    Used for FOM plot creation.

    To make array suitable for input use function similar to this:
    array = np.hstack((array_1, array_2, array_3, array_4, array_5, array_6)).reshape(-1,array_1.shape[0])

    Args:
        ND_array            :           input array of NxN dimensions.
        xlabel              :           x label list
        ylabel              :           y label list
        title               :           Title of plot
        xlabel_title        :           x label
        ylabel_title        :           y label

    Returns:
        None                :           empty return
    '''


    nx, ny = ND_array.shape

    indx, indy = np.arange(ny), np.arange(nx)
    x, y = np.meshgrid(indx, indy, indexing='ij')

    fig, ax = plt.subplots()
    ax.imshow(ND_array, interpolation="none")

    for xval, yval in zip(y.flatten(), x.flatten()):
        zval = ND_array[xval, yval]
        t = zval # format value with 1 decimal point
        c = 'k' if zval > 0.75 else 'w' # if dark-green, change text color to white
        ax.text(yval, xval, t, color=c, va='center', ha='center')


    xlabels = xlabel
    ylabels = ylabel

    ax.set_xticks(indx+0.5) # offset x/y ticks so gridlines run on border of boxes
    ax.set_yticks(indy+0.5)
    ax.grid(ls='-', lw=2)
    ax.set_xlabel(xlabel_title)
    ax.set_ylabel(ylabel_title)
    ax.set_title(title)

    # the tick labels, if you want them centered need to be adjusted in 
    # this special way.
    for a, ind, labels in zip((ax.xaxis, ax.yaxis), (indx, indy), 
                            (xlabels, ylabels)):
        a.set_major_formatter(ticker.NullFormatter())
        a.set_minor_locator(ticker.FixedLocator(ind))
        a.set_minor_formatter(ticker.FixedFormatter(labels))

    ax.xaxis.tick_top()

# It works! Functionalise
def scrape_FOM_data(data_path):
    '''
    Scrape all FOM data from a h5 file and plot the result with plot_2Dhist

    Args:
        data_path           :       data path leading to relevant h5 file

    Returns:
        None                :       empty return
    '''


    # collect data
    hold = pd.HDFStore(data_path)
    store = hold.keys()
    hold.close()

    # sanitise
    remove = [x.replace("/", "") for x in store]
    split = [(x.split("_")) for x in remove]

    # collect all unique first elements (n_iter)
    unique_0s = list(set([x[0] for i, x in enumerate(split)]))
    # and second elements
    unique_1s = list(set([x[1] for i, x in enumerate(split)]))

    # organise and add leading zero to column (1s)
    unique_0s_ = sorted(unique_0s, key=float)
    unique_1s_ = sorted(unique_1s, key=float)
    unique_1s_ = ["n_iter"] + unique_1s_
    
    # create pandas dataframe with these as the axis
    df = pd.DataFrame(columns = unique_1s_)
    
    # start adding rows babee
    for i in range(len(unique_0s_)):
    	df.loc[i] = [unique_0s_[i]] + list(np.full(shape=len(unique_1s), fill_value=np.nan))

    # set it as the index as well
    df.set_index('n_iter', inplace=True)
    
    # collect the data from each table in the h5 dataframe
    for i in range(len(store)):
        # reformat store data to allow for correct indexing
        remove = store[i].replace("/", "")
        split = remove.split("_")

        play_thing = pd.read_hdf(data_path,key=store[i])
        play_thing.set_index('Cut', inplace=True)
        fom_val = play_thing.loc['FOM_MAX - blob2_E_val (MeV)', 'Relative Efficiency']
            
        df.at[split[0], split[1]] = fom_val
	
    fom_list = []
    for i in range(len(df.index)):
        fom_list.append(df.loc[df.index[i]].to_numpy())
    
    fom_list = np.array(fom_list)
    #reshape into x,y array
    x_vals = (df.columns).to_list()
    y_vals = (df.index).to_list()
    
    # set nans to zeros
    fom_list[np.isnan(fom_list.astype(float))] = 0
    
    fom_list = np.round(fom_list.astype(float), decimals=2)
    
    plot_2Dhist(fom_list, x_vals, y_vals, title = 'FOM LPR', xlabel_title = 'number of iterations', ylabel_title = 'e_cut')
    


# this one differs, it allows you to scrape any data and make a 2D FOM plot
def scrape_any_data(data_path, string_1, string_2, plot_title):
    '''
    Similar to function 'scrape_FOM_data', 
    but works with any row and column from efficiency h5 file.

    Args:
        data_path           :           data path leading to relevant h5 file
        string_1            :           row of interest
        string_2            :           column of interest
        plot_title          :           plot title
    
    Returns:
        None                :           empty return
    '''

    # collect data
    hold = pd.HDFStore(data_path)
    store = hold.keys()
    hold.close()

    # sanitise
    remove = [x.replace("/", "") for x in store]
    split = [(x.split("_")) for x in remove]

    # collect all unique first elements (n_iter)
    unique_0s = list(set([x[0] for i, x in enumerate(split)]))
    # and second elements
    unique_1s = list(set([x[1] for i, x in enumerate(split)]))

    # organise and add leading zero to column (1s)
    unique_0s_ = sorted(unique_0s, key=float)
    unique_1s_ = sorted(unique_1s, key=float)
    unique_1s_ = ["n_iter"] + unique_1s_
    
    # create pandas dataframe with these as the axis
    df = pd.DataFrame(columns = unique_1s_)
    
    # start adding rows babee
    for i in range(len(unique_0s_)):
    	df.loc[i] = [unique_0s_[i]] + list(np.full(shape=len(unique_1s), fill_value=np.nan))

    # set it as the index as well
    df.set_index('n_iter', inplace=True)
    
    # collect the data from each table in the h5 dataframe
    for i in range(len(store)):
        # reformat store data to allow for correct indexing
        remove = store[i].replace("/", "")
        split = remove.split("_")

        play_thing = pd.read_hdf(data_path,key=store[i])
        play_thing.set_index('Cut', inplace=True)
        fom_val = play_thing.loc[str(string_1), str(string_2)]
            
        df.at[split[0], split[1]] = fom_val
	
    fom_list = []
    for i in range(len(df.index)):
        fom_list.append(df.loc[df.index[i]].to_numpy())
    
    fom_list = np.array(fom_list)
    #reshape into x,y array
    x_vals = (df.columns).to_list()
    y_vals = (df.index).to_list()
    
    # set nans to zeros
    fom_list[np.isnan(fom_list.astype(float))] = 0
    
    fom_list = np.round(fom_list.astype(float), decimals=2)
    
    plot_2Dhist(fom_list, x_vals, y_vals, title = str(plot_title), xlabel_title = 'number of iterations', ylabel_title = 'e_cut')



    # useful to normalize histograms
def get_weights(data, norm):
    '''
    Creates weights for normalising data for use in histograms.
    Source from Helena's functions

    Args:
        data            :       dataframe
        norm            :       normalise boolean

    Returns:
        weights         :       weights of the data

    '''
    if norm:
        return np.repeat(1.0/len(data), len(data))
    else:
        return np.repeat(1.0, len(data))

def energy_track_plots(tracks, title = "Low pressure track energies", limit = [0]):
    '''
    Plot the 2D histogram of number of tracks against track energy

    Args:
        tracks          :       dataframe of tracks
        title           :       title of plot
        limit           :       limits of the energy tracks
    
    Returns:
        None            :       empty return
    '''
    track_energy = tracks.energy
    track_no = tracks.numb_of_tracks
    # normalise
    weights = get_weights(track_energy, True)
    #plt.hist2d(track_energy, track_no, bins=(50, 20), cmin=0.001)
    plt.hist2d(track_energy, track_no, weights = weights, bins=(100, 11), cmin=0.0005)
    if limit == [0]:
        print("No limits applied")
    else:
        plt.xlim([0,1])
    plt.title(title)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Number of tracks')
    plt.colorbar()
    plt.show()

def process_data(path):
    '''
    Collects isaura data files from a defined folder, applies cuts and
    calculates FOM and efficiency plots.

    Args:
        path        :       path to folder containing h5 files

    Returns:
        None        :       empty return
    '''
    print("Opening files...")
    # load and unpack data, assume you're sitting in the PORT_XX folder
    data = load_data(str(folder_path) + 'isaura/') 
    tracks = data[0]
    particles = data[1]
    eventmap = data[2]


    # save raw histogram
    plot_hist(tracks, column = 'energy', output= False, binning = 65, title = "raw_hist",
            fill = True, data = False, save = True, save_dir = str(folder_path) + 'output/')


    print("Applying Cuts...")

    # remove low energy satellites first
    low_e_cut_tracks = remove_low_E_events(tracks)


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
    fiducial_rel = fiducial_track_cut_2(low_e_cut_tracks, lower_z = 20, upper_z=1195, r_lim = 472, verbose = False)

    fiducial_abs = fiducial_track_cut_2(tracks, lower_z = 20, upper_z=1195, r_lim = 472, verbose = True)

    # make efficiency calculation
    print("Fiducial track cut")
    print("==================")
    print("Relative Cut efficiency:")
    ef = cut_effic(fiducial_rel, low_e_cut_tracks)
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

    # apply cuts
    ovlp_rel = overlapping_cuts(one_track_rel)
    ovlp_abs = overlapping_cuts(tracks)


    cut_names.append("Blob overlap cuts")

    # relative
    print("Blob overlap cut")
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

    ecut_rel = energy_cuts(ovlp_rel)
    ecut_abs = energy_cuts(tracks)

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
            efficiencies.loc[len(efficiencies.index)] = ['pos_evt - all_evt', 0, len(ecut_rel), 0]
            efficiencies.loc[len(efficiencies.index)] = ['FOM_MAX - blob2_E_val (MeV)', 0, 0, 0]
            efficiencies.to_csv(str(folder_path) + 'output/efficiency.csv')
            print("No events left in ROI... jobs done!")
            return 0

            
        
    plot_hist(ecut_rel, column = 'energy', output= False, binning = 20, title = "cut_hist",
                fill = True, data = False, save = True, save_dir = str(folder_path) + 'output/', log = False)


    ###########################################################################################
    # EFFICIENCY CALCULATION OVER
    ###########################################################################################

    print("Calculating FOM")

    # collect positron events
    positron_events = positron_scraper(str(folder_path) + 'isaura/')
    pos_events = (np.unique(positron_events['event_id'].to_numpy()))*2

    # number of events that are positrons
    ecut_positron_df = ecut_rel[ecut_rel['event'].isin(pos_events)]
    ecut_no_positron_df = ecut_rel[~ecut_rel['event'].isin(pos_events)]
    cut_list = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    fom = true_fom_calc(ecut_positron_df, ecut_no_positron_df, cut_list)
    # sanitise
    fom = np.nan_to_num(fom)

    print("FOM values:")
    print(fom)

    # remove stupid values based on low statistics
    fom[fom > 10] = 0
    fom[fom < 0] = 0

    max_index = np.argmax(fom)


    efficiencies.loc[len(efficiencies.index)] = ['pos_evt - all_evt', len(ecut_positron_df), len(ecut_rel), 0]
    efficiencies.loc[len(efficiencies.index)] = ['FOM_MAX - blob2_E_val (MeV)', fom[max_index], cut_list[max_index], 0]
    
    efficiencies.to_csv(str(folder_path) + 'output/efficiency.csv')

    print("Jobs done!")

    # Save the data to a h5 file
    ecut_rel.to_hdf(str(folder_path) + 'output/post_cuts.h5', key='cut_data', mode = 'w')

###########################################################################################
# ERROR CALCULATIONS
###########################################################################################


def ratio_error(f, a, b, a_error, b_error):
    '''
    Error multiplication via quadrature

    Args:
        f       :       efficiency (%)
        a       :       events post-cut (be it signal or background)
        b       :       total events (likewise)
        a_error :       sqrt of events post-cut
        b_error :       sqrt of total events
    
    Returns:
        f_error :       cumulative error
    '''
    f_error = f*np.sqrt((a_error/a)**2 +(b_error/b)**2)
    return f_error


def fom_error(a, b, a_error, b_error):
    '''
    Produces error for figure of merit
    Derived in Joplin notes: 11/04/24
    
    Args:
        a           :       signal efficiency
        b           :       background acceptance
        a_error     :       signal error
        b_error     :       background error
    
    Returns:
        f_error     :       FOM error
    
    '''

    element_1 = np.square(a_error/np.sqrt(b))
    element_2 = np.square((b_error * a) / (2*(b**(3/2))))
    f_error = np.sqrt(element_1 + element_2)

    return f_error