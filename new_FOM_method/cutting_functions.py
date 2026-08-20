import pandas as pd
import numpy  as np


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


def apply_cuts(tracks, lower_z = 20, upper_z = 1195, r_lim = 472, lower_e = 1.5, upper_e = 1.7):
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
