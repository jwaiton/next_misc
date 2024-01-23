# A magical place for holding all the useful functions for
# analysing isaura data

import sys,os,os.path

sys.path.append("../../../")   # cite IC from parent directory
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

import scipy.special as special
from scipy.stats import skewnorm
from scipy.optimize import curve_fit

from scipy.integrate import quad


def cut_effic(df1, df2, verbose = False):
    '''
    Prints efficiency of cuts for singular cut
    df1 -> cut df
    df2 -> initial df
    '''
    length_1 = df1['event'].nunique()
    length_2 = df2['event'].nunique()
    efficiency = ((length_1/length_2)*100)
    

    if (verbose == True):
        print("Efficiency: {:.2f} %".format(efficiency))
        print(("Events in reduced dataframe: {}\nEvents in initial dataframe: {}").format(len(df1), len(df2)))

    return efficiency


def ratio_error(f, a, b):
    '''
    Error multiplication via quadrature

    f - efficiency (%)
    a - events post-cut
    b - total events
    a_error - sqrt of events post-cut
    b_error - sqrt of total events
    '''

    a_error = np.sqrt(a)
    b_error = np.sqrt(b)

    f_error = f*np.sqrt((a_error/a)**2
                                  +(b_error/b)**2)
    return f_error



def plot_hist(df, column = 'energy', binning = 20, title = "Energy plot", output = True, fill = True, label = 'default', x_label = 'energy (MeV)', range = 0, data = False, save_title = 'False', linestyle = 'solid', dens = False):
    '''
    Print a histogram of energy from our dataframe,.
    '''
    # for simplicity/readability, scoop out the relevant columns from the pandas dataframe.
    energy_vals = df[column].to_numpy()

    if (range==0):
        range = (np.min(energy_vals), np.max(energy_vals))

    # control viewing of hist
    if (fill == True):
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, range = range, linestyle = linestyle, density = dens)
    else:
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, histtype='step', linewidth = 2, range = range, linestyle = linestyle, density = dens)
    plt.title(title)
    plt.ylabel("events")
    plt.xlabel(x_label)

    if (save_title == 'False'):
        print("")
    else:
        plt.savefig(save_title)
        

    if (output==True):
        plt.show()
    if (data==True):
        return (cnts, edges, patches)
    else:
        return




def fiducial_track_cut_2(df, lower_z = 20, upper_z = 1195, r_lim = 472, verbose = False):
    '''
    Produces fiducial track cuts while removing all events that have outer fiducial tracks
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
    Remove events with more than one track
    THERE IS A COLUMN WITH THIS INFO IN IT, CALCULATING IT IS UNNECESSARY
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
    Remove all events with energy overlap != 0
    '''

    ovlp_remove = df[df['ovlp_blob_energy']==0]

    if (verbose==True):
        print("Removing overlapping blobs...")

    return ovlp_remove




def energy_cuts(df, lower_e = 1.5, upper_e = 1.7, verbose = False):
    '''
    Apply cuts around the relevant energy
    '''
    filt_e_df = df[(df['energy'] >= lower_e) & (df['energy'] <= upper_e)]

    if (verbose == True):
        print("Cutting energy events around {} & {} keV".format(lower_e, upper_e))

    return filt_e_df




def len_events(df):
    '''
    Returns the number of unique events as len(df) doesn't work in this case
    '''
    length_1 = df['event'].nunique()
    return length_1


# useful to normalize histograms
def get_weights(data, norm):
    '''
    Function useful for normalising histograms
    '''
    if norm:
        return np.repeat(1.0/len(data), len(data))
    else:
        return np.repeat(1.0, len(data))


def plot_volume_hists(df):
    '''
    Function used for plotting a dataframe's distribution throughout x,y,z
    '''
    plot_hist(df, column = 'x_min', output = False, label = 'x_min', fill = False, x_label = 'position (mm)')
    plot_hist(df, column = 'x_max', title = 'x plot for full volume', output = False, label = 'x_max', fill = False, x_label = 'position (mm)')
    plt.legend(loc='upper left')
    plt.show()

    plot_hist(df, column = 'y_min', output = False, label = 'y_min', fill = False, x_label = 'position (mm)')
    plot_hist(df, column = 'y_max', title = 'y plot for full volume', output = False, label = 'y_max', fill = False, x_label = 'position (mm)')
    plt.legend(loc='upper left')
    plt.show()

    plot_hist(df, column = 'z_min', output = False, label = 'z_min', fill = False, x_label = 'position (mm)')
    plot_hist(df, column = 'z_max', title = 'z plot for full volume', output = False, label = 'z_max', fill = False, x_label = 'position (mm)')
    plt.legend(loc='upper left')
    plt.show()

def read_MC_tracks(folder_path):
    '''
    Read in data from isaura events (MC and tracks)
    Checks the event mapping and adjusts if as expected
    '''

    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # NOTE Break this section up, its annoying like this.
    dfs = []
    df_trs = []
    df_ems = []
    # create massive dataframe with all of them
    for file in file_names:
        file_path = folder_path + file
        df = dstio.load_dst(file_path, 'Tracking', 'Tracks')
        dfs.append(df)

        # include MC particles (boooo takes ages)

        # collecting the correct components of the file, not exactly sure how this works
        df_ps = pd.read_hdf(file_path, 'MC/particles')

        # collecting event map
        df_em = mcio.load_eventnumbermap(file_path).set_index('nexus_evt')
        df_trs.append(df_ps)
        df_ems.append(df_em)

    tracks = pd.concat(dfs, axis=0, ignore_index=True)

    particles = pd.concat(df_trs, ignore_index=True)
    eventmap = pd.concat([dt for dt in df_ems])
    # create particle list also

    # check that the event map is as expected and modify to match MC to true data
    eventmap_reset = eventmap.reset_index()
    if not (eventmap_reset['nexus_evt'] * 2 == eventmap_reset['evt_number']).all():
        print("Event mapping no longer accurate between nexus and isaura events.\nPlease look at the eventmap object again")
    else:
        particles['event_id'] = particles['event_id'] * 2 
    
    # return 
    return (tracks, particles)


def remove_low_E_events(df, energy_limit = 0.05):
    '''
    Remove low energy tracks, add their energy back to the first
    track and then update 'numb_of_tracks' to be up to date
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


def apply_cuts_raw(tracks, e_low_cut = 0.05, fid_lower_z = 20, fid_upper_z = 1195, fid_r_lim = 472, e_lower = 1.35, e_upper = 1.9):
    '''
    Apply all relevant cuts and spit out dataframe at the end
    The most barebones version, no efficiency calcs, it just spits out the dataframe
    '''
    print("Start events: {}".format(tracks['event'].nunique()))

    # Low energy tracks, make this function work better. It seems busted currently
    low_e_cut_tracks = isa.remove_low_E_events(tracks, energy_limit = e_low_cut)
  
    # make fiducial cuts
    fiducial_rel = isa.fiducial_track_cut_2(low_e_cut_tracks, lower_z = fid_lower_z, upper_z=fid_upper_z, r_lim = fid_r_lim, verbose = False)

    # one track cuts 
    one_track_rel = isa.one_track_cuts(fiducial_rel, verbose = False)

    # overlapping blob cut
    ovlp_rel = isa.overlapping_cuts(one_track_rel)


    # energy cuts
    ecut_rel = isa.energy_cuts(ovlp_rel, lower_e = e_lower, upper_e = e_upper)

    print("End events: {}".format(ecut_rel['event'].nunique()))


    # return it
    return ecut_rel


def apply_all_cuts(tracks, verbose = False, low_e_cut_plot = 'False'):
    '''
    Apply all relevant cuts
    
    returns efficiency table and the fully cut data
    
    
    This function makes me physically ill,
    but was made in a rush.
    Please rework this, for the love of god
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

    low_e_cut_tracks = remove_low_E_events(tracks)


    # plot low e cut tracks if you want
    if (low_e_cut_plot == 'False'):
        print("")
    else:

        track_energy = low_e_cut_tracks.energy
        track_no = low_e_cut_tracks.numb_of_tracks
        # normalise
        weights = get_weights(track_energy, True)
        # clearing just in case
        plt.clf()
        # plot
        plt.hist2d(track_energy, track_no, weights = weights, bins=(100, tracks['numb_of_tracks'].max()), cmin=0.001)
        plt.title("Track energies: Low Pressure w/ low E cut: " + str(low_e_cut_plot))
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Number of tracks')
        plt.xlim([0,2.0])
        plt.colorbar()

        plt.savefig("trk_E_vs_no_trk_low_E.png")


        if (verbose == True):
            plt.show()


    # low energy cut
    cut_names.append("low E cuts")

    # make efficiency calculation
    if (verbose == True):
        print("Low E cut")
        print("==================")
        ef = cut_effic(low_e_cut_tracks, tracks, verbose = True)
    else:
        ef = cut_effic(low_e_cut_tracks, tracks)

    # add efficiency calculation
    rel_cut_effics.append(ef)
    cut_events.append(len_events(low_e_cut_tracks))
    abs_cut_effics.append(ef)



    cut_names.append("Fiducial Cuts")

    # make fiducial cuts
    fiducial_rel = fiducial_track_cut_2(low_e_cut_tracks, lower_z = 20, upper_z=1195, r_lim = 472, verbose = False)

    fiducial_abs = fiducial_track_cut_2(tracks, lower_z = 20, upper_z=1195, r_lim = 472, verbose = False)

    # make efficiency calculation
    if (verbose == True):
        print("Fiducial track cut")
        print("==================")
        print("Relative Cut efficiency:")
        ef = cut_effic(fiducial_rel, low_e_cut_tracks, verbose = True)
    else:
        ef = cut_effic(fiducial_rel, low_e_cut_tracks, verbose = False)
    rel_cut_effics.append(ef)
    cut_events.append(len_events(fiducial_rel))

    if (verbose == True):
        print('Absolute Cut efficiency:')
        ef = cut_effic(fiducial_abs, tracks, verbose = True)
    else:
        ef = cut_effic(fiducial_abs, tracks, verbose = False)
    abs_cut_effics.append(ef)


    # relative single track
    cut_names.append("One track cut")
    one_track_rel = one_track_cuts(fiducial_rel, verbose = False)

    # events are relative, as absolute efficiency lets you figure out events from the beginning# absolute
    one_track_abs = one_track_cuts(tracks)

    ## sanity check here
    #print(len_events(one_track_rel), len_events(one_track_abs), len_events(tracks))

    # relative
    if (verbose == True):
        print("Single track cut")
        print("================")
        print("Relative Cut efficiency:")
        ef = cut_effic(one_track_rel, fiducial_rel, verbose = True)
    else:
        ef = cut_effic(one_track_rel, fiducial_rel, verbose = False)
    rel_cut_effics.append(ef)
    cut_events.append(len_events(one_track_rel))

    # absolute
    if (verbose == True):
        print("Absolute Cut efficiency:")
        ef = cut_effic(one_track_abs, tracks, verbose = True)
    else:
        ef = cut_effic(one_track_abs, tracks, verbose = False)
    abs_cut_effics.append(ef)


    # apply cuts
    ovlp_rel = overlapping_cuts(one_track_rel)
    ovlp_abs = overlapping_cuts(tracks)

    cut_names.append("Blob overlap cuts")

    
    # relative
    if (verbose == True):
        print("Blob overlap cut")
        print("================")
        print("Relative Cut efficiency:")
        ef = cut_effic(ovlp_rel, one_track_rel, verbose = True)
    else:
        ef = cut_effic(ovlp_rel, one_track_rel, verbose = False)
    rel_cut_effics.append(ef)
    cut_events.append(len_events(ovlp_rel))


    # absolute
    if (verbose == True):
        print("Absolute Cut efficiency:")
        ef = cut_effic(ovlp_abs, tracks, verbose = True)
    else:    
        ef = cut_effic(ovlp_abs, tracks, verbose = False)
    abs_cut_effics.append(ef)


    # energy cuts
    ecut_rel = energy_cuts(ovlp_rel)
    ecut_abs = energy_cuts(tracks)

    cut_names.append("Energy cuts")

    # relative
    if (verbose == True):
        print("Energy cut")
        print("================")
        print("Relative Cut efficiency:")
        ef = cut_effic(ecut_rel, ovlp_rel, verbose = True)
    else:
        ef = cut_effic(ecut_rel, ovlp_rel, verbose = False)
    rel_cut_effics.append(ef)
    cut_events.append(len_events(ecut_rel))


    # absolute
    if (verbose == True):    
        print("Absolute Cut efficiency:")
        ef = cut_effic(ecut_abs, tracks, verbose = True)
    else:
        ef = cut_effic(ecut_abs, tracks, verbose = False)
    abs_cut_effics.append(ef)


    efficiencies = pd.DataFrame({'Cut': cut_names,
                             'Relative Efficiency': rel_cut_effics,
                             'Relative Events': cut_events,
                             'Single Cut Efficiency': abs_cut_effics
                             })

    if (verbose == True):
        display(efficiencies)
        print("Single Cut Efficiency: each cut on the original unmodified data set\nRelative Efficiency: each cut efficiency wrt the previous cut")


    # return it
    return (ecut_rel, efficiencies)



def process_efficiencies(folder_paths, folder_titles, verbose = True, low_e_plot = True):
    '''
    Function that read in MC and Track data from isaura files,
    Applies cuts and calculates the efficiencies, then plots
    track energy vs # of tracks and saves the main plots

    Also collects the histogram outputs for the energy distributions
    '''

    # make the directory to hold plots in
    directory_name = 'efficiency_study_plots'
    
    # list that holds all the efficiency tables as a tuple of (Name,efficiency_table)
    efficiency_objects = []
    # hold all the histogram outputs for the energy resolution
    hist_objects = []

    # loop over data
    for i in range(len(folder_paths)):

        
        ############################################################
        print("Processing {}...\n".format(folder_titles[i]))
        ############################################################


        # read in, this data will be rewritten each time
        tracks, particles = read_MC_tracks(folder_paths[i])

        if (verbose == True):
            print("================")
            print("  VERBOSE MODE  ")
            print("================")
            display(tracks.head())
            display(particles.head())
            display(tracks.tail())
            display(particles.tail())


        ############################################################
        print("Data read. Producing plots...\n")
        ############################################################


        # first we create track energies vs no of tracks plot
        track_energy = tracks.energy
        track_no = tracks.numb_of_tracks
        # normalise
        weights = get_weights(track_energy, True)
        # bins set up to match number of tracks
        plt.hist2d(track_energy, track_no, weights = weights, bins=(100, tracks['numb_of_tracks'].max()), cmin=0.001)
        plt.xlim([0,2.0])
        plt.title("Track energies LPR: Precuts - " + str(folder_titles[i]))
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Number of tracks')
        plt.colorbar()

        # save figure
        if not os.path.isdir(directory_name + '/' + folder_titles[i]):
            os.mkdir(directory_name + '/' + folder_titles[i])

        plt.savefig(directory_name + '/' + folder_titles[i] + '/' +str("trk_E_vs_no_trk.png"))
        
        # show plot
        if (verbose==True):
            plt.show()
        else:
            plt.close()


        ############################################################
        print("Applying efficiency cuts...\n")
        ############################################################
        # Efficiency calculation


        # dont look at this function if you want to retain brain cells
        if (low_e_plot == True):
            ecut_rel, efficiencies = apply_all_cuts(tracks, verbose = False, low_e_cut_plot = folder_titles[i])
            # move file
            os.replace("trk_E_vs_no_trk_low_E.png", directory_name + '/' + folder_titles[i] + '/' + "trk_E_vs_no_trk_low_E.png")
        else:
            ecut_rel, efficiencies = apply_all_cuts(tracks, verbose = False, low_e_cut_plot = 'False')
            

        # append with name as tuple
        efficiency_objects.append([folder_titles[i],efficiencies])

        ############################################################
        print("Producing more plots...\n")
        ############################################################
        
        name_save = str(directory_name) + '/' + str(folder_titles[i]) + '/' +str("Energy_plot.png")

        plt.close()
        hist_o = plot_hist(ecut_rel, binning = 50, title = 'Energy Plots' + str(folder_titles[i]), save_title = name_save, data = True)

        hist_objects.append([folder_titles[i], hist_o])
        if (verbose==True):
            plt.show()
        else:
            plt.close()
        

        # clean all variables here
        del hist_o
        del tracks
        del particles
        del ecut_rel
        del track_energy
        del track_no
        del weights



    ############################################################
    print("Job's done!")
    ############################################################

    return (efficiency_objects, hist_o)


        


def default_fit(data, bins = 75, verbose = False):
    '''
    Apply the expected fit to the data here,
    works directly with fom_calc
    '''

    # find minima and maxima
    evalues = data['energy'].to_numpy()
    e_low = np.min(evalues) 
    e_high = np.max(evalues)

    # collect heights from histogram of energy
    hist, edges, patches = plot_hist(data, binning = bins, output = False, data = True)
    plt.clf()
    # convert edges to centres
    centres = shift_to_bin_centers(edges)


    # collect assumptions
    p1 = [1, 1.58, 0.006, 5]
    gauss_bck_labels = ['a', 'mu', 'sigma', 'C']

    # fit function
    popt, pcov = curve_fit(gauss_bck, centres, hist, p1, maxfev = 5000000)

    if (verbose == True):
        plot_fit(gauss_bck, centres, popt, gauss_bck_labels)
        plot_fit(gauss, centres, popt[:-1], gauss_bck_labels[:-1], lgnd = 'Gauss Fit', colour = 'cyan', popt_text = False)
        plot_fit(bck, centres, [popt[-1]], gauss_bck_labels[-1], lgnd = 'Background Fit', colour = 'blue', popt_text = True)
        
        plot_hist(data, binning = bins, output = False, data = True, label='Data')
        plt.legend()
        plt.show()
        print_parameters(popt, pcov, gauss_bck_labels)

    # centres passed through to ensure we get the correct even numbers
    return (popt, pcov, gauss_bck_labels, centres)
        



def fom_calc(data, cut_list, no_pos_data = pd.DataFrame({'A' : []}), binning = 75, verbose = False):
    '''
    produces a figure of merit list based
    on cuts to specific categories and their
    consequent fits

    no_pos_data related to data with no positrons in it (background).
    Useful to visualise for 
    '''

    if ((verbose == True) and not (no_pos_data.empty)):
        blob_positron_plot(data, no_pos_data)
    # Take the initial, no blob2 cut values for ns and nb
    output = default_fit(data, bins = binning, verbose = True)

    popt = output[0]
    pcov = output[1]
    gauss_bck_labels = output[2]
    centres = output[3]

    # take bin widths to calculate number of events
    bin_width = centres[1] - centres[0]

    # signal is integration of function over the space
    ns0 = quad(gauss, emin, emax, args = tuple(popt[:-1]))/bin_width
    nb0 = quad(bck, emin, emax, args = popt[-1])/bin_width

    if (verbose == True):
        print('ns0      = {}'.format(ns0[0]))
        print('nb0      = {}'.format(nb0[0]))
        print("total    = {:.0f}".format(ns0[0]+nb0[0]))
        print("Event no = {}".format(len(data.index)))


    # create all the lists for fom
    e = []
    b = []
    fom = []

    for i in range(len(cut_list)):
        print("")
        print("")
        print("")
        
        print("==========================")
        print("        CUT {} MeV       ".format(cut_list[i]))
        print("==========================")
        # remove blob 2 values below value on cut_list
        data = data[(data['eblob2'] > cut_list[i])]
        if not (no_pos_data.empty):
            no_pos_data = no_pos_data[(no_pos_data['eblob2'] > cut_list[i])]
        # apply fit to the new data 
        if (verbose == True):
            output = default_fit(data, bins = binning, verbose = True)
        else:
            output = default_fit(data, bins = binning, verbose = False)
        
        # collect values
        popt = output[0]
        pcov = output[1]
        gauss_bck_labels = output[2]
        centres = output[3]

        # take bin widths to calculate number of events
        bin_width = centres[1] - centres[0]

        # collect ns and nb
        ns = quad(gauss, emin, emax, args = tuple(popt[:-1]))/bin_width
        nb = quad(bck, emin, emax, args = popt[-1])/bin_width

        if (verbose == True):
            print('ns - {}'.format(ns[0]))
            print('nb - {}'.format(nb[0]))
            print("total = {:.0f}".format(ns[0]+nb[0]))
            print("Event no = {}".format(len(data.index)))
        # produce fom value (DISCREETLY NOW AS IT ISNT WORKING)
        e_check     = ns[0]/ns0[0]
        b_check     = nb[0]/nb0[0]
        fom_check   = e_check/np.sqrt(b_check)
        

        print('\ne_i: {}\nb_i: {}\nfom: {}'.format(e_check, b_check, fom_check))

        e.append(e_check)
        b.append(b_check)
        fom.append(fom_check)

        if ((verbose == True) and not (no_pos_data.empty)):
            blob_positron_plot(data, no_pos_data)
        
    # that should be it? i dont expect this to work first time, but lets test it!
    return fom






def true_fom_calc(p_data, no_p_data, cut_list, verbose = False):
    '''
    produces a figure of merit list based
    on cuts to specific categories and their
    consequent fits

    '''

    # create deep copies for safety
    pos_data = p_data.copy(deep = True)
    no_pos_data = no_p_data.copy(deep = True)

    if (verbose == True):
        blob_positron_plot(pos_data, no_pos_data)
    # Take the initial, no blob2 cut values for ns and nb
    ns0 = len(pos_data.index)
    nb0 = len(no_pos_data.index)
    # create all the lists for fom
    e = []
    b = []
    fom = []

    for i in range(len(cut_list)):
        
        # remove blob 2 values below value on cut_list
        pos_data = pos_data[(pos_data['eblob2'] > cut_list[i])]
        no_pos_data = no_pos_data[(no_pos_data['eblob2'] > cut_list[i])]


        # apply fit to the new data 
        if (verbose == True):
            print("Signal events: {}\nBackground events: {}\n FOM: {}".format())
        
        # collect number of signal events vs number of backgrounds, which you know 
        ns = len(pos_data.index)
        nb = len(no_pos_data.index)

        # produce fom value
        e.append(ns/ns0)
        b.append(nb/nb0)
        fom.append(e[i]/np.sqrt(b[i]))

        if (verbose == True):
            blob_positron_plot(pos_data, no_pos_data)
        
    # that should be it? i dont expect this to work first time, but lets test it!
    return fom


def positron_scraper(data):
    '''
    pass through data, get back only the positron data :)
    '''
    return data[data['particle_name'] == 'e+']



def read_positron_scraper(data_path, save = False):
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


def double_event_id(df):
    '''
    Doubles the event_id of a dataframe passed to it.
    This is useful for when the nexus mapping and the IC mapping is different
    by exactly a half.
    '''

    df['event_id'] = df['event_id'] * 2

    return df