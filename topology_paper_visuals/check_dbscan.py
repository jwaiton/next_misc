import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import sys, os

sys.path.append('/home/e78368jw/Documents/NEXT_CODE/IC/')
os.environ['ICTDIR']='/home/e78368jw/Documents/NEXT_CODE/IC/'

from invisible_cities.cities.components import track_blob_info_creator_extractor
from invisible_cities.io.hits_io        import hits_from_df
plt.rcParams.update({
    # Use LaTeX for text rendering
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],

    # Font sizes (match your LaTeX doc's font size)
    "font.size": 12*2,
    "axes.titlesize": 16*2,
    "axes.labelsize": 14*2,
    "xtick.labelsize": 12*2,
    "ytick.labelsize": 12*2,
    "legend.fontsize": 12*2,

    # Figure size — match LaTeX text width
    # For A4 with default margins: ~6.3in wide
    "figure.figsize": (5.9, 5.9),  # golden ratio height

    # Line/marker quality
    "lines.linewidth": 1.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def cluster_tagger(df_hits      : pd.DataFrame
                  , *
                  , min_samples : int
                  , scale_xy    : float
                  , scale_z     : float
                  ) -> pd.DataFrame:
    """
    This function groups the input DataFrame by 'event' and applies the
    `tag_hits_in_event` function to each event's group of hits.

    Parameters
    ----------
    df_hits : pd.DataFrame
        DataFrame with hit information. Must contain 'X', 'Y', 'Z', and 'event'.
    min_samples, scale_xy, scale_z :
        See `tag_hits_in_event`

    Returns
    -------
    pd.DataFrame
        The input DataFrame with an added 'cluster' column indicating the
        cluster label for each hit (-1 for noise).
    """
    if df_hits.empty:
        return df_hits.assign(cluster=pd.Series(dtype=int))
    # expecting one df
    df_clustered = tag_hits_in_event(df_hits, min_samples = min_samples, scale_xy = scale_xy, scale_z = scale_z)

    return df_clustered.set_index(df_hits.index)

def tag_hits_in_event(event_hits   : pd.DataFrame
                     , *
                     , min_samples : int
                     , scale_xy    : float
                     , scale_z     : float
                     ) -> pd.DataFrame:
    """
    Applies DBSCAN clustering to a DataFrame containing hits from a single event.
    Hits coordinates are scaled to account for the anisotropy of the detector geometry.
    A 'cluster' column is added to the group with the resulting labels.

    Parameters
    ----------
    event_hits  : pd.DataFrame
        DataFrame with hits from a single event. Must contain 'X', 'Y', 'Z' columns.
    min_samples : int
        Minimum number of samples required to form a dense region (cluster).
        This includes the point itself.
    scale_xy    : float
        Scaling factor to apply to the XY coordinates before clustering.
    scale_z     : float
        Scaling factor to apply to the Z coordinate before clustering.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with a 'cluster' column added.
    """
    coords = event_hits[['X', 'Y', 'Z']].to_numpy()
    # A proper scaling leads to hits being separeted
    # by a distance of 1 in the DBSCAN metric space
    coords[:, :2] /= scale_xy
    coords[:, 2]  /= scale_z

    # eps parameter is fixed to a value a bit higher of √3
    # to retain diagonal neighbours in the same cluster
    labels = DBSCAN(eps=1.8, min_samples=min_samples).fit_predict(coords)
    event_hits['cluster'] = labels

    return event_hits


def raw_plotter(q, evt, pitch = 15.55, title = None, plot_lims = None, blob_locations = None, blob_energies = None, text_pos = None, y_axis_label = True, save = False):
    '''
    just plots the hits, nothing smart
    '''

    fig, ax = plt.subplots(figsize = (6.3,5))
    xx = np.arange(q.X.min() - pitch*2, q.X.max() + pitch*2, pitch)
    zz = np.sort(q.Z.unique())
    ax.hist2d(q.X, q.Z, bins=[xx, zz], weights=q.Q, cmin=0.0000001, zorder=1);
    ax.set_xlabel('X (mm)');
    if y_axis_label:
        ax.set_ylabel('Z (mm)');

    if blob_locations is not None:
        # assume ((x1,y1),(x2,y2))
        cx = blob_locations[0][0]
        cy = blob_locations[0][1]
        circle_B1 = patches.Circle((blob_locations[0][0], blob_locations[0][1]), radius=35,
                          facecolor='none', edgecolor='red', linestyle = 'dashed',
                          linewidth=2, zorder=2, label = 'B1')
        if blob_energies is not None:
            if text_pos is None:
                ax.text(cx + 35 + 0.15, cy + 35 +  0.15, f'E: {blob_energies[0]:.2f} MeV',  fontsize=14*2, zorder=3)
            else:
                ax.text(cx + text_pos[0][0], cy + text_pos[0][1], f'E: {blob_energies[0]:.2f} MeV',  fontsize=14*2, zorder=3)



        cx = blob_locations[1][0]
        cy = blob_locations[1][1]

        circle_B2 = patches.Circle((blob_locations[1][0], blob_locations[1][1]), radius=35,
                          facecolor='none', edgecolor='magenta', linestyle = 'dotted',
                          linewidth=2, zorder=2, label = 'B2')
        if blob_energies is not None:
            if text_pos is None:
                ax.text(cx + 35 + 0.15, cy + 35 + 0.15, f'E: {blob_energies[1]:.2f} MeV',  fontsize=14*2, zorder=3)
            else:
                ax.text(cx + text_pos[1][0], cy + text_pos[1][1], f'E: {blob_energies[1]:.2f} MeV',  fontsize=14*2, zorder=3)


        ax.add_patch(circle_B1)
        ax.add_patch(circle_B2)
    if title is None:
        plt.title("rebinned in Z")
    else:
        plt.title(f'{title}')
    if plot_lims is not None:
        ax.set_xlim(plot_lims[0][0], plot_lims[0][1])
        ax.set_ylim(plot_lims[1][0], plot_lims[1][1])
    ax.legend()
    if save:
        plt.savefig(f"plots/{title.replace(' ', '_')}_{evt}.pdf", bbox_inches='tight', pad_inches = 0.05)
        plt.savefig(f"plots/{title.replace(' ', '_')}_{evt}.png", bbox_inches='tight', pad_inches = 0.05)
    plt.show()


files = glob.glob('data/*.h5')
#for f in files[::-1]:
#    try:
#        x = pd.read_hdf(f, 'RECO/Events')
#        for evt, df in x.groupby('event'):
#            if (df.Ec.sum() > 1.4) & (df.Ec.sum() < 1.7):
#                if df.npeak.nunique() == 1:
#                    print(f"file {f.split('/')[-1:]} evt {evt}")
#                    raw_plotter(df, evt, title = f'{evt}')
#    except Exception as e:
#        print(e)

#evt = 31774
#file = '0043'

## nice background
#evt = 46509
#file = '0063'

# nice signal
#evt  = 35197
#file = '0048'

#background shenans
#evt  = 51857
#file = '0070'

#evt  = 58409
#file = '0079'

#evt  = 44073
#file = '0060'

evt = 45375
file = '0061'

x = pd.read_hdf(f'data/run_15589_{file}_ldc1_230725_thekla.h5', 'RECO/Events')

x = x[x.event == evt]


raw_plotter(x, evt)

x_clustered = cluster_tagger(x, min_samples = 5, scale_xy = 15.55, scale_z = 4)
x_cluster_0 = x_clustered[x_clustered['cluster'] == 0]
x_cluster_1 = x_clustered[x_clustered['cluster'] == 1]


raw_plotter(x_cluster_0, evt, title = f'{x_cluster_0.Ec.sum()} MeV - cluster 0, {len(x_cluster_0)} hits')

raw_plotter(x_cluster_1, evt, title = f'{x_cluster_1.Ec.sum()} MeV - cluster 1, {len(x_cluster_1)} hits')

exit()
# insert ep
x['Ep'] = x['Ec']
#x = x[x.Z > 800]

def hitc_from_df(hits : pd.DataFrame):
    hitcs = hits_from_df(hits)
    if len(hitcs) == 0:
        return HitCollection(0, 0, []) # dummy HitCollection
    assert len(hitcs) == 1
    for hitc in hitcs.values():
        return hitc

saved = True
if not saved:

    x_hitc = hitc_from_df(x)
    # extract blob information
    t_b = track_blob_info_creator_extractor(vox_size = [15., 15., 15.],
                                            strict_vox_size = False,
                                            energy_threshold = 0.01,
                                            min_voxels = 3,
                                            blob_radius = 35.,
                                            scan_radius = 40.,
                                            max_num_hits = 100000000000)
    df_out, track_hits, _ = t_b(x_hitc)

    df_out.to_hdf('data/track_info_signal.h5', 'Tracking/Tracks')
else:
    df_out = pd.read_hdf('data/track_info_signal.h5', 'Tracking/Tracks')
raw_plotter(x, evt, title = f'Candidate signal event',
            plot_lims = ((150, 400), (810, 1060)),
            blob_locations = ((df_out['blob1_x'].values[0],df_out['blob1_z'].values[0]),
                              (df_out['blob2_x'].values[0],df_out['blob2_z'].values[0])),
            blob_energies = (df_out['eblob1'].values[0], df_out['eblob2'].values[0]),
            text_pos = ((35, 27), (0, -47)))
            #blob_locations = ((327.8, 1014.0), (246.7, 852.1)))



file = '0005'
evt  = 4103
#evt = 2409
#file = '0003'
y = pd.read_hdf(f'data/run_15589_{file}_ldc1_230725_thekla.h5', 'RECO/Events')

y = y[y.event == evt]
y = y[y.Z > -100]
y['Ep'] = y['Ec']
if not saved:

    y_hitc = hitc_from_df(y)
    df_out, track_hits, _ = t_b(y_hitc)

    df_out.to_hdf('data/track_info_background.h5', 'Tracking/Tracks')
else:
    df_out = pd.read_hdf('data/track_info_background.h5', 'Tracking/Tracks')
raw_plotter(y, evt, title = f'Candidate background event',
            blob_locations = ((df_out['blob1_x'].values[0],df_out['blob1_z'].values[0]),
                             (df_out['blob2_x'].values[0],df_out['blob2_z'].values[0])),
            plot_lims = ((-480, -50), (900, 1200)),
            blob_energies = (df_out['eblob1'].values[0], df_out['eblob2'].values[0]),
            text_pos = ((0, 40), (-42, -55)))


