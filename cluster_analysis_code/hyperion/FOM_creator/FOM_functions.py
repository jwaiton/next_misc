'''

AN OVERARCHING PYTHON SCRIPT THAT IM GOING TO FORCE ALL MY FUNCTIONS INTO ONE WAY OR ANOTHER
IM NOT COPYING AND PASTING THE 'LOAD_DATA' FUNCTION EVER AGAIN. I REFUSE.

Any time I need to use a function, I'm adding it to this stupidly long list of functions.

i have resultantly made multiple versions of this 'functions' file. Stupid stupid stupid stupid.
'''

from concurrent.futures import ProcessPoolExecutor


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

# timekeeping
from tqdm import tqdm

from scipy.integrate import quad


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


def load_data_open(folder_path, node_path):
    '''
    Load in multiple h5 files and produce dataframes corresponding to whatever node the user wants ('MC/particles, etc'), and their corresponding
    eventmap. Just dont have it be a dstio.load_dst, then it'll break

    Args:
        folder_path     :       path to folder of h5 files
        node_path       :       path within h5 to data
    Returns:
        (tracks,        :       tracks dataframe
        particles,      :       MC particle information dataframe
        eventmap)       :       eventmap for MC -> Tracks
    '''
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.h5')]
    
    # remove any files that dont end in h5

    # NOTE Break this section up, its annoying like this.
    dfs = []

    i = 0
    end = len(file_names)
    # create massive dataframe with all of them
    for file in file_names:
        file_path = folder_path + file
        df_ps = pd.read_hdf(file_path, node_path)

        dfs.append(df)
        # include MC particles (boooo takes ages)

        # collecting the correct components of the file, not exactly sure how this works

        i += 1

        if (i%50 == 0):
            print(i)

    tracks = pd.concat(dfs, axis=0, ignore_index=True)

    # create particle list also

    return (tracks)

def load_single_file(file_path):
    '''
    Load data from a single h5 file and produce dataframes for /Tracking/Tracks, /MC/Particles, and their corresponding eventmap.

    Args:
        file_path       :       str
                                Path to the h5 file to be loaded.

    Returns:
        tracks_df       :       pandas.DataFrame
                                DataFrame containing the /Tracking/Tracks data.
        
        particles_df    :       pandas.DataFrame
                                DataFrame containing the /MC/particles data, with the 'event_id' column modified.

        eventmap_df     :       pandas.DataFrame
                                DataFrame containing the event map, indexed by 'nexus_evt'.
    '''

    tracks_df = dstio.load_dst(file_path, 'Tracking', 'Tracks')
    #particles_df = pd.read_hdf(file_path, 'MC/particles')
    #eventmap_df = mcio.load_eventnumbermap(file_path).set_index('nexus_evt')
    
    # Modify particles data
    #particles_df['event_id'] = particles_df['event_id'] * 2
    
    return tracks_df#, particles_df, eventmap_df

def load_data_fast(folder_path):
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
        results = list(executor.map(load_single_file, file_paths))
    
    # Separate the results into respective lists
    #tracks_list = particles_list, eventmap_list = zip(*results)
    tracks_list = results
    # Concatenate all the dataframes at once
    tracks = pd.concat(tracks_list, axis=0, ignore_index=True)
    #particles = pd.concat(particles_list, ignore_index=True)
    #ventmap = pd.concat(eventmap_list, ignore_index=True)

    return tracks# particles




def collate_ports(path_array):
    '''
    Collect individual ports and merge the information

    Args:
        path_array          :           an array of folder paths to h5 files
                                        respective of the multiple ports

    
    Returns:
        array               :           output of collective ports
    '''

    # strip array except from port information
    port_id = [x.split('PORT_')[1][:2] for x in path_array]

    for i in range(len(path_array)):
        print("Porting {}".format(path_array[i]))
        if (i==0):
            tracks = (load_data(path_array[i]))[0]
            # add on the column for port ID
            tracks['PORT'] = str(port_id[i])
        else:
            nu_tracks = (load_data(path_array[i]))[0]
            print("Tracks: {}".format(len_events(nu_tracks)))
            # multiply the events numbers to avoid overlap that doesnt work sipshit.
            nu_tracks['event'] = nu_tracks['event'] #* (i+1)
            nu_tracks['PORT'] = str(port_id[i])
            tracks = tracks.append(nu_tracks)
        try:
            unique_events = tracks.drop_duplicates(subset=['event', 'PORT'])
            print("Done! Tracks available: {}".format(unique_events.shape[0]))
        except:
            print("Done! Tracks available: {}".format(len_events(tracks)))
            
    
    return tracks

def cut_purity(df1, signal, verbose = False):
    '''
    Produce effiency of a single cut by comparison values in two databases wrt
    a selection of all the signal events.
    
    Purity = total number of signal events post-cut / total events in selected sample
    
    Warning! Ensure your signal event id's match the formatting of your
    other dataframes (usually by multiplying them by two)
    
    Args:
        df1         :       dataframe 1 (post-cut)
        signal      :       dataframe of exclusively signal events
        verbose      :       verbose boolean
    
    Returns:
        purity      :       purity of cuts
    '''
    # collect event numbers
    signal_evts = signal.event_id.unique()
    df1_evts    = df1.event.unique()
    
    # select all signal events that are in initial dataframe via intersection
    post_cut_signal_evts = np.intersect1d(df1_evts, signal_evts)
    
    # calculate purity by differing number of signal events left in data
    purity = len(post_cut_signal_evts) / len(df1_evts) * 100
    print("Purity: {:.2f} %".format(purity))

    
    
    if (verbose == True):
        print(f'Purity of {len(post_cut_signal_evts)} / {len(df1_evts)} = {purity}')
    
    return purity
    
    
    
    
    


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


def fiducial_track_cut_2(df, lower_z = 20, upper_z = 1195, r_lim = 472, verbose = False, ID = 'event'):
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
    low_list = (z_df_low[ID].to_numpy())
    up_list = (z_df_up[ID].to_numpy())
    r_list = (r_df[ID].to_numpy())

    # apply the filter to remove all events that fall in outer fiduc
    df1 = df[~df[ID].isin(low_list)]
    df2 = df1[~df1[ID].isin(up_list)]
    df3 = df2[~df2[ID].isin(r_list)]

    if (verbose == True):
        print("Cutting events around fiducial volume related to:\nZ range between {} and {}\nRadius range < {}".format(lower_z, upper_z, r_lim))


    return df3




def one_track_cuts(df, verbose = False, ID = 'event'):
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
    event_counts = df.groupby(ID).size()
    #print(event_counts[:5]) # showing that you see how many 
                            #  trackIDs there are per event
    one_track = event_counts[event_counts == 1].index

    # filter dataframe
    one_track_events = df[df[ID].isin(one_track)]
    

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




def energy_cuts(df, lower_e = 1.5, upper_e = 1.7, verbose = False, ID = 'event'):
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
    filt_e_df = (df[(df['energy'] >= lower_e) & (df['energy'] <= upper_e)])[ID].unique()
    
    
    
    

    if (verbose == True):
        print("Cutting energy events around {} & {} keV".format(lower_e, upper_e))

    return df[df[ID].isin(filt_e_df)]


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

def from_file_positron(file):
    """
    Function that takes an individual file and spits out a dataframe
    with exclusively positron events

    Args:
        file        :       .h5 file with path
    
    Returns:
        pos_df      :       positron dataframe
    """


    # load in file
    MC_df = pd.read_hdf(file, 'MC/particles')
    evt_map = (mcio.load_eventnumbermap(file).set_index('nexus_evt'))

    pos_data = MC_df[MC_df['particle_name'] == 'e+']

        
    return pos_data


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


def positron_ports(path_array):
    '''
    Collect positron events from multiple ports.
    Bespoke, use with care.
    '''

    for i in range(len(path_array)):
        print("Loading positrons from {}".format(path_array[i]))
        if (i==0):
            posi = positron_scraper(path_array[i])
            # multiply the event numbers to match track values
            posi['event_id'] = posi['event_id'] * ((i+1)*2)
        else:
            nu_posi = positron_scraper(path_array[i])
            print("Positron events: {}".format(len_events(nu_posi, tag = 'event_id')))
            # multiple the event numbers to avoid overlap, the *2 is to match it with the tracking values
            nu_posi['event_id'] = nu_posi['event_id'] * ((i+1)*2)
            posi = posi.append(nu_posi)
        print("Port finished! Tracks available: {}".format(len_events(posi, tag = 'event_id')))
    
    return posi

def numb_of_signal_events(data_path, fid = False, lower_z = 20, upper_z = 1170, r_lim = 415):
    '''
    Scans through all files, collects number of signal events available.
    Fiducial tag exists, but currently doesn't work (the MC exists everywhere, 
    so fiducial cuts don't mean anything)
    '''

    try:
        file_names = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f)) and f.endswith('.h5')]
    except:
        print("File path incorrect, please state the correct file path\n(but not any particular folder!)")

    event_id_list = []
    total_id_list = []

    for file in tqdm(file_names):
        double_escape_IDs = []
        file_path = data_path + file
        # read in file
        #data = dstio.load_dst(file_path, 'RECO', 'Events')
        df_ps = pd.read_hdf(file_path, 'MC/particles')

        # collect fiducial info
        if (fid == True):
            true_info = mcio.load_mchits_df(file_path).reset_index()

        event_list = (df_ps['event_id'].unique())
        total_id_list.append(event_list)

        pos_df = from_file_positron(file_path)
        signal_id = pos_df['event_id'].to_numpy()
        for i in range(len(signal_id)):
            
            # select specific event
            signal_data = pos_df[pos_df['event_id'] == signal_id[i]]
            #display(signal_data)
            df_ps_data = df_ps[df_ps['event_id'] == signal_id[i]]
            mother_id_pos = signal_data.particle_id.to_numpy()[0]
            positron_children = df_ps_data[df_ps_data['mother_id'] == mother_id_pos]
            #display(positron_children)
            # collect the annihilation gammas
            annihilation_gamma_id = positron_children[positron_children['creator_proc'] == 'annihil'].particle_id.to_numpy()
            #display(annihilation_gamma_id)
            for j in range(len(annihilation_gamma_id)):

                # check fiducial limits
                #if (fid == True):
                #    fid_evt = true_info[true_info['event_id'] == signal_id[i]]
                #    fid_evt = fid_evt[fid_evt['label'] == 'ACTIVE']

                #    zMin = np.min(fid_evt.z.to_numpy())
                #    zMax = np.max(fid_evt.z.to_numpy())

                #    r    = np.sqrt((fid_evt.x)**2 + (fid_evt.y)**2)

                #    rMax = np.max(r)

                #    # if outside fiducial (at any point), break
                #    if ((zMin < lower_z) or (zMax > upper_z) or(rMax > r_lim)):
                #        break
                #    else:
                #        continue




                # check the two gammas children, they better not exist
                gamma_children = (df_ps_data[df_ps_data['mother_id'] == annihilation_gamma_id[j]])
                #display(gamma_children)
                # check that no events occur from the children within ACTIVE
                gamma_active_1 = gamma_children[gamma_children['initial_volume'] == 'ACTIVE']
                gamma_active_2 = gamma_children[gamma_children['final_volume'] == 'ACTIVE']
                frames = [gamma_active_1, gamma_active_2]
                gamma_active = pd.concat(frames)
                if len(gamma_active.index) == 0:
                    double_escape_IDs.append(signal_id[i])
                else:
                    #print("Event {} not double-photo, remove...".format(signal_id[i]))
                    # kill it
                    if (j == 0):
                        break
                    elif (j == 1):
                        # remove previous double_escape_ID event from list
                        double_escape_IDs.remove(signal_id[i])



        double_escape_IDs = np.unique(np.array(double_escape_IDs))
        event_id_list.append(double_escape_IDs)
    
    # flatten lists
    flat_double_escape = [j for sub in event_id_list for j in sub]
    flat_total = [j for sub in total_id_list for j in sub]

    print("{}/{} events with double photo-escape".format(len(flat_double_escape), len(flat_total)))
    print("")
    print("{:.2f}% of events within file are signal events\n      within our energy region".format((len(flat_double_escape)/len(flat_total))*100))
    return (flat_double_escape, flat_total)


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

def apply_cuts_purity(tracks, true_tracks, lower_z = 20, upper_z = 1195, r_lim = 472, lower_e = 1.5, upper_e = 1.7, ports = False):
    '''
    Applies all known cuts, returns dataframe and efficiency table.
    Highly bespoke function, use with care.

    NOTE: Does not include satellite track removal

    Args:
        tracks          :       dataframe of particle tracks
        true_tracks     :       Events selected explicitly as signal
        lower_z         :       lower z-bound for fiducial cut
        upper_z         :       upper z-bound for fiducial cut
        r_lim           :       radial bound for fiducial cut
        lower_e         :       lower bound for energy cut
        upper_e         :       upper bound for energy cut
        ports           :       Multiple port boolean
                                need to provide a unique ID (p_evt)
                                if true
    
    Returns:
        (ecut_rel,      :       dataframe with output of track cuts
        efficiencies)           efficiency table
    '''
    # port boolean check
    if (ports == False):
        ID = 'event'
    elif (ports == True):
        ID = 'p_evt'
    else:
        raise TypeError('Only booleans allowed for parameter port')
    
    # Efficiency and purity calculation
    cut_names = []
    cut_events = []
    
    rel_cut_effics = []
    
    abs_cut_effics = []
    purity         = []
    
    

    # no cuts
    cut_names.append("No cuts")
    # number of events
    cut_events.append(len_events(tracks, tag = ID))
    
    rel_cut_effics.append(100)
    abs_cut_effics.append(100)
    
    purity.append(cut_purity(tracks, true_tracks))
    
    #####################################################################
    #####################################################################

    # fiducial cuts
    cut_names.append("Fiducial Cuts")

    # make fiducial cuts
    fiducial_rel = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = False, ID = ID)

    fiducial_abs = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = True, ID = ID)

    # make efficiency calculation
    print("Fiducial track cut")
    print("==================")
    print("Relative Cut efficiency:")
    ef = cut_effic(fiducial_rel, tracks)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(fiducial_rel, true_tracks))
    

    cut_events.append(len_events(fiducial_rel, tag = ID))

    print('Absolute Cut:')
    ef = cut_effic(fiducial_abs, tracks)
    abs_cut_effics.append(ef)

    #####################################################################
    #####################################################################

    cut_names.append("One track cut")
    one_track_rel = one_track_cuts(fiducial_rel, verbose = False, ID = ID)

    # events are relative, as absolute efficiency lets you figure out events from the beginning# absolute
    one_track_abs = one_track_cuts(tracks, ID = ID)

    # relative
    print("One track cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(one_track_rel, fiducial_rel)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(one_track_rel, true_tracks))


    cut_events.append(len_events(one_track_rel, tag = ID))

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
    purity.append(cut_purity(ovlp_rel, true_tracks))


    cut_events.append(len_events(ovlp_rel, tag = ID))
    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(ovlp_abs, tracks)
    abs_cut_effics.append(ef)

    
    #####################################################################
    #####################################################################

    ecut_rel = energy_cuts(ovlp_rel, lower_e, upper_e, ID = ID)
    ecut_abs = energy_cuts(tracks, lower_e, upper_e, ID = ID)

    cut_names.append("Energy cuts")

    # relative
    print("Energy cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(ecut_rel, ovlp_rel)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(ecut_rel, true_tracks))

    
    cut_events.append(len_events(ecut_rel, tag = ID))


    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(ecut_abs, tracks)
    abs_cut_effics.append(ef)

    #####################################################################
    #####################################################################
    # reminder that this only works due to 1-track cut being applied
    
    # blob 2 cuts > 0.26 MeV
    blobcut_rel = ecut_rel[(ecut_rel['eblob2'] > 0.26)]
    blobcut_abs = tracks[(tracks['eblob2'] > 0.26)]
    
    cut_names.append("Blob2 cut > 0.26 MeV")
    
    # relative
    print("Blob 2 cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(blobcut_rel, ecut_rel)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(blobcut_rel, true_tracks))

    
    cut_events.append(len_events(blobcut_rel, tag = ID))


    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(blobcut_abs, tracks)
    abs_cut_effics.append(ef)

    
    
    information = pd.DataFrame({'Cut': cut_names,
                             'Relative Efficiency': rel_cut_effics,
                             'Relative Events': cut_events,
                             'Single Cut Efficiency': abs_cut_effics,
                             'Purity': purity
                             }) 
    
  # adding exception in for when there's no data in ecut_rel
    if (len(ecut_rel.index) == 0):
            print("No events left in ROI... jobs done!")
    return (ecut_rel, information)
    
    
def apply_cuts(tracks, lower_z = 20, upper_z = 1195, r_lim = 472, lower_e = 1.5, upper_e = 1.7, ports = False, overlap_flag = True):
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
        ports           :       Multiple port boolean
                                need to provide a unique ID (p_evt)
                                if true
    
    Returns:
        (ecut_rel,      :       dataframe with output of track cuts
        efficiencies)           efficiency table
    '''
    # port boolean check
    if (ports == False):
        ID = 'event'
    elif (ports == True):
        ID = 'p_evt'
    else:
        raise TypeError('Only booleans allowed for parameter port')
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
    cut_events.append(len_events(tracks, tag = ID))


    #####################################################################
    #####################################################################

    # fiducial cuts
    cut_names.append("Fiducial Cuts")

    # make fiducial cuts
    fiducial_rel = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = False, ID = ID)

    fiducial_abs = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = True, ID = ID)

    # make efficiency calculation
    print("\nFiducial track cut")
    print("==================")
    print("Relative Cut efficiency:")
    ef = cut_effic(fiducial_rel, tracks)
    rel_cut_effics.append(ef)
    

    cut_events.append(len_events(fiducial_rel, tag = ID))

    print('Absolute Cut efficiency:')
    ef = cut_effic(fiducial_abs, tracks)
    abs_cut_effics.append(ef)



    #####################################################################
    #####################################################################

    cut_names.append("One track cut")
    one_track_rel = one_track_cuts(fiducial_rel, verbose = False, ID = ID)

    # events are relative, as absolute efficiency lets you figure out events from the beginning# absolute
    one_track_abs = one_track_cuts(tracks, ID = ID)

    # relative
    print("\nOne track cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(one_track_rel, fiducial_rel)
    rel_cut_effics.append(ef)

    cut_events.append(len_events(one_track_rel, tag = ID))

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

    cut_events.append(len_events(ovlp_rel, tag = ID))
    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(ovlp_abs, tracks)
    abs_cut_effics.append(ef)


    #####################################################################
    #####################################################################

    ecut_rel = energy_cuts(ovlp_rel, lower_e, upper_e, ID = ID)
    ecut_abs = energy_cuts(tracks, lower_e, upper_e, ID = ID)

    cut_names.append("Energy cuts")

    # relative
    print("\nEnergy cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(ecut_rel, ovlp_rel)
    rel_cut_effics.append(ef)

    cut_events.append(len_events(ecut_rel, tag = ID))


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



def apply_cuts_many_tracks(tracks, true_tracks, lower_z = 20, upper_z = 1195, r_lim = 472, lower_e = 1.5, upper_e = 1.7, ports = False):
    '''
    Applies all known cuts, returns dataframe and efficiency table.
    Highly bespoke function, use with care.

    NOTE: Does not include satellite track removal

    Args:
        tracks          :       dataframe of particle tracks
        true_tracks     :       Events selected explicitly as signal
        lower_z         :       lower z-bound for fiducial cut
        upper_z         :       upper z-bound for fiducial cut
        r_lim           :       radial bound for fiducial cut
        lower_e         :       lower bound for energy cut
        upper_e         :       upper bound for energy cut
        ports           :       Multiple port boolean
                                need to provide a unique ID (p_evt)
                                if true
    
    Returns:
        (ecut_rel,      :       dataframe with output of track cuts
        efficiencies)           efficiency table
    '''
    # port boolean check
    if (ports == False):
        ID = 'event'
    elif (ports == True):
        ID = 'p_evt'
    else:
        raise TypeError('Only booleans allowed for parameter port')
    
    # Efficiency and purity calculation
    cut_names = []
    cut_events = []
    
    rel_cut_effics = []
    
    abs_cut_effics = []
    purity         = []
    
    

    # no cuts
    cut_names.append("No cuts")
    # number of events
    cut_events.append(len_events(tracks, tag = ID))
    
    rel_cut_effics.append(100)
    abs_cut_effics.append(100)
    
    purity.append(cut_purity(tracks, true_tracks))
    
    #####################################################################
    #####################################################################

    # fiducial cuts
    cut_names.append("Fiducial Cuts")

    # make fiducial cuts
    fiducial_rel = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = False, ID = ID)

    fiducial_abs = fiducial_track_cut_2(tracks, lower_z, upper_z, r_lim, verbose = True, ID = ID)

    # make efficiency calculation
    print("Fiducial track cut")
    print("==================")
    print("Relative Cut efficiency:")
    ef = cut_effic(fiducial_rel, tracks)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(fiducial_rel, true_tracks))
    

    cut_events.append(len_events(fiducial_rel, tag = ID))

    print('Absolute Cut:')
    ef = cut_effic(fiducial_abs, tracks)
    abs_cut_effics.append(ef)

    
    #####################################################################
    #####################################################################
    
    
    cut_names.append("One track cut")
    one_track_rel = one_track_cuts(fiducial_rel, verbose = False, ID = ID)

    # events are relative, as absolute efficiency lets you figure out events from the beginning# absolute
    one_track_abs = one_track_cuts(tracks, ID = ID)

    # relative
    print("One track cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(one_track_rel, fiducial_rel)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(one_track_rel, true_tracks))


    cut_events.append(len_events(one_track_rel, tag = ID))

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
    purity.append(cut_purity(ovlp_rel, true_tracks))


    cut_events.append(len_events(ovlp_rel, tag = ID))
    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(ovlp_abs, tracks)
    abs_cut_effics.append(ef)

    
    #####################################################################
    #####################################################################

    ecut_rel = energy_cuts(ovlp_rel, lower_e, upper_e, ID = ID)
    ecut_abs = energy_cuts(tracks, lower_e, upper_e, ID = ID)

    cut_names.append("Energy cuts")

    # relative
    print("Energy cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(ecut_rel, ovlp_rel)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(ecut_rel, true_tracks))

    
    cut_events.append(len_events(ecut_rel, tag = ID))


    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(ecut_abs, tracks)
    abs_cut_effics.append(ef)

    #####################################################################
    #####################################################################
    # reminder that this only works due to 1-track cut being applied
    
    # blob 2 cuts > 0.26 MeV
    blobcut_rel = ecut_rel[(ecut_rel['eblob2'] > 0.26)]
    blobcut_abs = tracks[(tracks['eblob2'] > 0.26)]
    
    cut_names.append("Blob2 cut > 0.26 MeV")
    
    # relative
    print("Blob 2 cut")
    print("================")
    print("Relative Cut efficiency:")
    ef = cut_effic(blobcut_rel, ecut_rel)
    rel_cut_effics.append(ef)
    purity.append(cut_purity(blobcut_rel, true_tracks))

    
    cut_events.append(len_events(blobcut_rel, tag = ID))


    # absolute
    print("Absolute Cut efficiency:")
    ef = cut_effic(blobcut_abs, tracks)
    abs_cut_effics.append(ef)

    
    
    information = pd.DataFrame({'Cut': cut_names,
                             'Relative Efficiency': rel_cut_effics,
                             'Relative Events': cut_events,
                             'Single Cut Efficiency': abs_cut_effics,
                             'Purity': purity
                             }) 
    
  # adding exception in for when there's no data in ecut_rel
    if (len(ecut_rel.index) == 0):
            print("No events left in ROI... jobs done!")
    return (ecut_rel, information)
    
    


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
    fom_erro = np.nan_to_num(fom[1])
    fom = np.nan_to_num(fom[0])

    print("ns, nb")
    print(ns)
    print(nb)
    

    print("FOM values:")
    print(fom)
    print("Errors")
    print(fom_erro)

    # remove stupid values based on low statistics
    fom[fom > 10] = 0
    fom[fom < 0] = 0

    max_index = np.argmax(fom)
    # prep output for efficiencies
    positron_events = len(ecut_positron_df)
    fom_max = fom[max_index]
    blob_val = cut_list[max_index]

    if (plot == True):
        plt.errorbar(cut_list, fom, yerr = fom_erro)
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

    plot_2Dhist(fom_list, x_vals, y_vals, title = str(plot_title), xlabel_title = 'e_cut', ylabel_title = 'number of iterations')



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

def energy_track_plots(tracks, title = "Low pressure track energies", limit = [0], xbins = 100, ybins = 11, weight = True, cm = 1):
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
    #weights = get_weights(track_energy, weight)
    #plt.hist2d(track_energy, track_no, bins=(50, 20), cmin=0.001)
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(16, 16))
    if cm == 1:
        hist1 = axes.hist2d(track_energy, track_no, bins=(xbins, ybins), density=True)
    else:
        hist1 = axes.hist2d(track_energy, track_no, bins=(xbins, ybins), density=True, cmin = cm)
    fig.colorbar(hist1[3], ax = axes)
    #plt.hist2d(track_energy, track_no, weights = weights, bins=(xbins, ybins), cmin=0.0005)
    if limit == [0]:
        print("No limits applied")
    else:
        plt.xlim(limit)
    plt.title(title)
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Number of tracks')
    #plt.colorbar()
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

###########################################################################################
# FITTING FUNCTIONS!!!
###########################################################################################

def bck_func(x, nb, tau):
    '''
    Function describing the background, an exponential with scaling from Nb and tau
    '''

    return nb*np.exp(-x/tau)

def bck_func_no_N(x, tau):
    return np.exp(-x/tau)
    


def skewnorm_func(x, a, mu, sigma):
    return skewnorm.pdf(x, a, loc = mu, scale = sigma)

def norm_func(x, a, mu, sigma):
    # a is just carried over for swapability with skewnorm
    return norm.pdf(x, loc = mu, scale = sigma)

def error_func(x, mu, sigma):
    pas = (x - mu)/(np.sqrt(2)*sigma)
    return special.erfc(pas)



def sig_func(x, ns, a, mu, sigma, C1, C2):
    
    return ns * (skewnorm_func(x, a, mu, sigma) + C1*error_func(x, mu, sigma) + C2)


def sig_func_no_N(x, a, mu, sigma, C1, C2):
    
    return (skewnorm_func(x, a, mu, sigma) + C1*error_func(x, mu, sigma) + C2)


def sig_bck_func(x, ns, a, mu, sigma, C1, C2, nb, tau):

    return bck_func(x, nb, tau) + sig_func(x, ns, a, mu, sigma, C1, C2)


def gauss_no_N(x, mu, sigma):
    numer = np.square(x - mu)
    denom = 2*np.square(sigma)

    return np.exp(-numer/denom)

# create gaussian initially for testing purposes
def gauss(x, a, mu, sigma):
    numer = np.square(x - mu)
    denom = 2*np.square(sigma)

    return a*np.exp(-numer/denom)


def gauss_norm(x, a, mu, sigma):
    numer = np.square(x - mu)
    denom = 2*np.square(sigma)
    norm = (np.sqrt(2*np.pi) * sigma)


    return (a*np.exp(-numer/denom))/ norm

def gauss_bck_norm(x, a, mu, sigma, C):
    numer = np.square(x - mu)
    denom = 2*np.square(sigma)
    norm = (np.sqrt(2*np.pi) * sigma)

    return (a*np.exp(-numer/denom) / norm) + C    


def bck(x, C):
    return np.full_like(x, C)

def gauss_bck(x, a, mu, sigma, C):
    numer = np.square(x - mu)
    denom = 2*np.square(sigma)


    return a*np.exp(-numer/denom) + C


# and a print function
def print_parameters(popt,pcov,labels):
    '''
    Prints fitting parameters

    :param popt:        Fit parameters
    :param pcov:        Fit covariances
    :param labels:      Labels of parameters
    '''
    print('===============================')
    print("        Fitting output      ")
    print('===============================')
    for i in range(len(popt)):
        print("{}: {:.4f} \u00B1 {:.4f}".format(labels[i], popt[i], np.sqrt(pcov[i][i]) )) # taking diagonal covariances as errors
    print('===============================')
    return 0

 
# plot parameters for fit, similar to plot_hist
def plot_fit(function, x, popt, popt_label, output = False, colour = 'red', x_counts = 100000, lgnd = 'Fit', popt_text = True, linestyle = "solid"):
    '''
    plots a fit based on individual points and a function
    plots across a more continuous space, to reduce weird artifacting for low X numbers
    '''

    # take much more continuous x axis
    x_min = np.min(x)
    x_max = np.max(x)

    x_plot = np.linspace(x_min, x_max, num = x_counts, endpoint = True)

    y = function(x_plot, *popt)
    plt.plot(x_plot, y, label = lgnd, color = colour, linestyle = linestyle)
    
    # create locations to put the text relative to the scale of the figure
    percentage_hor = 0.01
    percentage_vert = 0.95
    x_loc = np.min(x) + (np.max(x) - np.min(x))*percentage_hor
    y_loc = np.min(y) + (np.max(y) - np.min(y))*percentage_vert
    # reasonable gap for separation, based on scale
    gap = (np.max(y) - np.min(y)) * 0.05

    if (popt_text == False):
        for i in range(len(popt)):

            plt.text(x_loc, y_loc - gap*i, str(popt_label[i]) + ": " + str(round(popt[i], 5)), verticalalignment='top', horizontalalignment='left')

    if (output == True):
        plt.show()
    else:
        return






###########################################################################################
# NEW FOM FUNCTIONS - 26/04/24!!!
###########################################################################################




def histogram_fit(fnc, sig_data, binning, p0, fit_labels, bounds = []):
    '''
    fit a function from histogram data, return the fitting parameters
    '''


    # Use positron data to collect the C1 and C2 values from the signal fit
    s_cnts, s_edges, s_patches = plot_hist(sig_data, binning = binning, log = False, data = True)

    
    s_centres = shift_to_bin_centers(s_edges)

    # FIT
    if (bounds == []):
        return curve_fit(fnc, s_centres, s_cnts, p0, maxfev = 500000)
    else:
        return curve_fit(fnc, s_centres, s_cnts, p0, maxfev = 500000, bounds = bounds)

def fom_calc_MC(cut_data, positron_data, cut_list, binning = 80, verbose = False):
    '''
        calculate FOM via fitting using MC information for C1 and C2
        start the cut list at non-zero. eg cut_list = [0.1, 0.2, ...]
    '''
    # preset some parameters for sanity purposes
    emin = 1.5
    emax = 1.7


    # select only events in which events have positrons
    sig_data = cut_data[cut_data['event'].isin(positron_data['event_id'].to_numpy())]
    bck_data = cut_data[~cut_data['event'].isin(positron_data['event_id'].to_numpy())]


    print("Obtaining C1 and C2")
    #####            C1 AND C2 ACQUISITION          #####
    # p0 is apriori
    p0 = ([1, 1, 1.58, 0.3, 0.8, 0])
    fit_labels = ['B1', 'A', 'mu', 'sigma', 'C1', 'C2']
    
    # fit the histogram
    s_popt, s_pcov = histogram_fit(sig_func, sig_data, binning, p0, fit_labels)

    
    if (verbose == True):
        print("=========================== SIGNAL FIT ============================")
        plot_fit(sig_func, np.linspace(emin, emax, 1000), s_popt, fit_labels)
        plot_hist(sig_data, binning = 80, title='Signal fit', log = False)
        plt.show()
        print_parameters(s_popt, s_pcov, fit_labels)
    
    # Set C1 and C2
    C1 = s_popt[4]
    C2 = s_popt[5]

    # C1 and C2 control
    if (C1 < 0):
        C1 = 0
    if (C2 < 0):
        C2 = 0

    print("C1: {}, C2: {}".format(C1, C2))



    #####           MU AND SIGMA ACQUISITION            #####

    # apriori
    g_p0 = [500, 1.6, 0.01]
    g_labels = ['A', 'mu', 'sigma']

    # collect histogram information

    #cnt, edges, patches = plot_hist(cut_data, binning = binning, log = False, data = True)
    # fit
    #g_popt, g_pcov = curve_fit(gauss, centres, cnts, g_p0, maxfev = 500000)
    g_popt, g_pcov = histogram_fit(gauss, cut_data, binning, g_p0, g_labels)
    # set mu and sigma
    mu      = g_popt[1]
    sigma   = g_popt[2]

    print("mu: {}, sigma: {}".format(mu, sigma))

    if (verbose == True):
        print("=========================== GAUSSIAN FIT ============================")
        plot_fit(gauss, np.linspace(emin, emax, 1000), g_popt, g_labels)
        plot_hist(cut_data, binning = 80, title='Gauss fit', log = False)
        plt.show()
        print_parameters(g_popt, g_pcov, g_labels)


    #####          NS AND NB ACQUISITION                #####

    fixed_sig_bck_func = lambda x, ns, a, nb, tau: sig_bck_func(x, ns, a, mu, sigma, C1, C2, nb, tau)

    # apriori
    sb_p0 = [400, 1, 20, 0.1]
    sb_labels = ['ns', 'a', 'nb', 'tau']

    # fit
    sb_popt, sb_pcov = histogram_fit(fixed_sig_bck_func, cut_data, binning, sb_p0, sb_labels)
    #sb_popt, sb_pcov = curve_fit(fixed_sig_bck_func, centres, cnts, sb_p0, maxfev = 500000)
    # take bin widths to calculate number of events
    _, edges, _ =plot_hist(cut_data, binning = binning, log = False, data = True)
    bin_width = edges[1] - edges[0]
    ns0 = quad(sig_func, emin, emax, args = (sb_popt[0],sb_popt[1], mu, sigma, C1, C2))/bin_width
    nb0 = quad(bck_func, emin, emax, args = (sb_popt[2], sb_popt[3]))/bin_width

    if (verbose == True):

        print("=========================== FULL FIT ============================")
        plot_fit(fixed_sig_bck_func, np.linspace(emin, emax, 1000), sb_popt, sb_labels)
        plot_hist(cut_data, binning = 80, title='Full fit', log = False)
        plt.show()
        print_parameters(sb_popt, sb_pcov, sb_labels)

        print('ns0      = {}'.format(ns0[0]))
        print('nb0      = {}'.format(nb0[0]))
        print("total    = {:.0f}".format(ns0[0]+nb0[0]))
        print("Event no = {}".format(len(cut_data.index)))
    
    
    # create list for fom values
    e       = []
    b       = []
    ns_l      = []
    nb_l      = []
    fom     = []
    fom_err = []
    e_err = []
    b_err = []

    ns_l.append(ns0[0])
    nb_l.append(nb0[0])

    # wipe variables to stop variable bleed over
    del g_popt, g_pcov, mu, sigma, sb_popt, sb_pcov, bin_width

    if (verbose == True):
        print("=========================== ====================== ===========================")
        print("=========================== BLOB 2 CUT STARTS HERE ===========================")
        print("=========================== ====================== ===========================")

    for i in range(len(cut_list)):

        print("Applying cut {} MeV".format(cut_list[i]))

        blob_data = cut_data[(cut_data['eblob2'] > cut_list[i])]

        # collect gaussian peak
        g_popt, g_pcov = histogram_fit(gauss, blob_data, binning, g_p0, g_labels)
        # set mu and sigma
        mu      = g_popt[1]
        sigma   = g_popt[2]

        if (verbose == True):
            print("=========================== GAUSSIAN FIT ============================")
            plot_fit(gauss, np.linspace(emin, emax, 1000), g_popt, g_labels)
            plot_hist(blob_data, binning = 80, title='Gauss fit', log = False)
            plt.show()
            print_parameters(g_popt, g_pcov, g_labels)


        # collect nb and ns
        sb_popt, sb_pcov = histogram_fit(fixed_sig_bck_func, blob_data, binning, sb_p0, sb_labels, bounds = ([0, -np.inf, 0, -np.inf],[np.inf, np.inf, np.inf, np.inf]))
        # take bin widths to calculate number of events
        _, edges, _ =plot_hist(blob_data, binning = binning, log = False, data = True, output = False)
        bin_width = edges[1] - edges[0]
        ns = quad(sig_func, emin, emax, args = (sb_popt[0],sb_popt[1], mu, sigma, C1, C2))/bin_width
        nb = quad(bck_func, emin, emax, args = (sb_popt[2], sb_popt[3]))/bin_width
        ns_l.append(ns[0])
        nb_l.append(nb[0])
        if (verbose == True):

            print("=========================== FULL FIT ============================")
            plt.clf()
            plot_fit(fixed_sig_bck_func, np.linspace(emin, emax, 1000), sb_popt, sb_labels, lgnd='Full fit')
            plot_fit(bck_func, np.linspace(emin, emax, 1000), sb_popt[-2:], sb_labels[-2:], lgnd='Background fit', colour = 'yellow')#, linestyle = 'dashed')

            # collect all sb_vales
            s_popt = [sb_popt[0], sb_popt[1], mu, sigma, C1, C2]
            s_labels = ['ns', 'a', 'mu', 'sigma', 'C1', 'C2']
            plot_fit(sig_func, np.linspace(emin, emax, 1000), s_popt, s_labels, lgnd='Signal fit', colour= 'green')#, linestyle = 'dashed')
            
            
            plot_hist(blob_data, binning = 80, title='Full fit', log = False, label = 'Data')
            plt.legend()
            plt.show()
            print_parameters(sb_popt, sb_pcov, sb_labels)

            print('ns - {}'.format(ns[0]))
            print('nb - {}'.format(nb[0]))
            print("total = {:.0f}".format(ns[0]+nb[0]))
            print("Event no = {}".format(len(blob_data.index)))
        
        e_check = ns[0]/ns0[0]
        b_check = nb[0]/nb0[0]
        fom_check = e_check/np.sqrt(b_check)

        e.append(e_check)
        b.append(b_check)
        fom.append(fom_check)

        # errors for fom
        e_err.append(ratio_error(e[i],ns[0],ns0[0],np.sqrt(ns[0]),np.sqrt(ns0[0])))
        b_err.append(ratio_error(b[i],nb[0],nb0[0],np.sqrt(nb[0]),np.sqrt(nb0[0])))
        fom_err.append(fom_error(e[i], b[i], e_err[i], b_err[i]))

        if (verbose == True):
            print('fom - {:.2f} ± {:.2f}'.format(fom_check, fom_err[i]))
            print('e - {:.2f} ± {:.2f}'.format(e_check, e_err[i]))
            print('b - {:.2f} ± {:.2f}'.format(b_check, b_err[i]))

        # wipe variables here
        del blob_data, g_popt, g_pcov, mu, sigma, sb_popt, sb_pcov, ns, nb, bin_width, e_check, b_check, fom_check
    
    return (fom, fom_err, ns_l, nb_l)

