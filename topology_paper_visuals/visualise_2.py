import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import plot_info
import numpy as np
import sys, os

sys.path.append('/home/e78368jw/Documents/NEXT_CODE/IC/')
os.environ['ICTDIR'] = '/home/e78368jw/Documents/NEXT_CODE/IC/'

from invisible_cities.cities.components import track_blob_info_creator_extractor
from invisible_cities.io.hits_io        import hits_from_df


# ----------------------------------------------------------------------
# drawing: operates on an existing axes, creates no figure
# ----------------------------------------------------------------------

def draw_event(ax, q, pitch=15.55, title=None, plot_lims=None,
               blob_locations=None, blob_energies=None, text_pos=None,
               y_axis_label=True):
    '''
    Draw a single X-Z hit display onto an existing axes.

    ax             - target axes (created by the caller)
    q              - per-event dataframe with X, Z, Q columns
    plot_lims      - ((xmin, xmax), (zmin, zmax))
    blob_locations - ((x1, z1), (x2, z2))
    blob_energies  - (E1, E2) in MeV
    text_pos       - ((dx1, dz1), (dx2, dz2)) offsets for the energy labels
    '''
    xx = np.arange(q.X.min() - pitch*2, q.X.max() + pitch*2, pitch)
    zz = np.sort(q.Z.unique())
    ax.hist2d(q.X, q.Z, bins=[xx, zz], weights=q.Q, cmin=0.0000001, zorder=1)

    ax.set_xlabel('X (mm)')
    if y_axis_label:
        ax.set_ylabel('Z (mm)')

    if blob_locations is not None:
        for i, (edge, style, lbl) in enumerate(
                [('red', 'dashed', 'B1'), ('magenta', 'dotted', 'B2')]):
            cx, cy = blob_locations[i]
            ax.add_patch(patches.Circle((cx, cy), radius=35,
                                        facecolor='none', edgecolor=edge,
                                        linestyle=style, linewidth=2,
                                        zorder=2, label=lbl))
            if blob_energies is not None:
                dx, dy = text_pos[i] if text_pos is not None else (35.15, 35.15)
                ax.text(cx + dx, cy + dy,
                        f'E: {blob_energies[i]:.2f} MeV', zorder=3)

    if title is not None:
        ax.set_title(title)

    if plot_lims is not None:
        ax.set_xlim(*plot_lims[0])
        ax.set_ylim(*plot_lims[1])

    ax.legend()


# ----------------------------------------------------------------------
# figure construction: fixed panel size via fixed_axes_figure
# ----------------------------------------------------------------------

def plot_event_pair(q_left, q_right, left_kwargs, right_kwargs,
                    ax_width=2.9, ax_height=2.6, wspace=0.65,
                    left=0.75, right=0.15, top=0.35, bottom=0.6,
                    savepath=None, show=True):
    '''
    Side-by-side event displays (e.g. signal vs background). Both
    panels are exactly ax_width x ax_height inches.

    Note: no bbox_inches='tight' on save - that would recrop each
    figure to its own content and undo the fixed-size guarantee.
    The left/right/top/bottom margins handle padding instead.
    '''
    fig, (axL, axR) = plot_info.fixed_axes_figure(
        1, 2, ax_width=ax_width, ax_height=ax_height, wspace=wspace,
        left=left, right=right, top=top, bottom=bottom)

    draw_event(axL, q_left,  **left_kwargs)
    draw_event(axR, q_right, **right_kwargs)

    if savepath:
        fig.savefig(savepath + '.pdf')
        fig.savefig(savepath + '.png', dpi=300)
    if show:
        plt.show()

    return fig, (axL, axR)


def plot_event_single(q, ax_width=2.9, ax_height=2.6,
                      left=0.75, right=0.15, top=0.35, bottom=0.6,
                      savepath=None, show=True, **draw_kwargs):
    '''
    Single-panel version. Panel size matches plot_event_pair when
    called with the same ax_width/ax_height, so a solo figure sits
    consistently alongside the paired one on the page.
    '''
    fig, ax = plot_info.fixed_axes_figure(
        1, 1, ax_width=ax_width, ax_height=ax_height,
        left=left, right=right, top=top, bottom=bottom)

    draw_event(ax, q, **draw_kwargs)

    if savepath:
        fig.savefig(savepath + '.pdf')
        fig.savefig(savepath + '.png', dpi=300)
    if show:
        plt.show()

    return fig, ax


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def hitc_from_df(hits: pd.DataFrame):
    hitcs = hits_from_df(hits)
    if len(hitcs) == 0:
        return HitCollection(0, 0, [])  # dummy HitCollection
    assert len(hitcs) == 1
    for hitc in hitcs.values():
        return hitc


def make_extractor():
    return track_blob_info_creator_extractor(vox_size=[15., 15., 15.],
                                             strict_vox_size=False,
                                             energy_threshold=0.01,
                                             min_voxels=3,
                                             blob_radius=35.,
                                             scan_radius=40.,
                                             max_num_hits=100000000000)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

# setup the style
plot_info.apply_style(scale_factor=1/1.35)

saved = True

# ---- signal event ----------------------------------------------------
# other candidates tried:
#   evt = 31774, file = '0043'
#   evt = 46509, file = '0063'   # nice background
#   evt = 35197, file = '0048'   # nice signal
sig_evt  = 51857
sig_file = '0070'

x = pd.read_hdf(f'data/run_15589_{sig_file}_ldc1_230725_thekla.h5', 'RECO/Events')
x = x[x.event == sig_evt]
x['Ep'] = x['Ec']

if not saved:
    t_b = make_extractor()
    df_sig, _, _ = t_b(hitc_from_df(x))
    df_sig.to_hdf('data/track_info_signal.h5', 'Tracking/Tracks')
else:
    df_sig = pd.read_hdf('data/track_info_signal.h5', 'Tracking/Tracks')

# ---- background event ------------------------------------------------
bkg_evt  = 4103
bkg_file = '0005'

y = pd.read_hdf(f'data/run_15589_{bkg_file}_ldc1_230725_thekla.h5', 'RECO/Events')
y = y[y.event == bkg_evt]
y = y[y.Z > -100]
y['Ep'] = y['Ec']

if not saved:
    t_b = make_extractor()
    df_bkg, _, _ = t_b(hitc_from_df(y))
    df_bkg.to_hdf('data/track_info_background.h5', 'Tracking/Tracks')
else:
    df_bkg = pd.read_hdf('data/track_info_background.h5', 'Tracking/Tracks')

# ---- side-by-side figure ---------------------------------------------
plot_event_pair(
    x, y,
    left_kwargs=dict(
        title='Candidate signal event',
        plot_lims=((150, 400), (810, 1080)),
        blob_locations=((df_sig['blob1_x'].values[0], df_sig['blob1_z'].values[0]),
                        (df_sig['blob2_x'].values[0], df_sig['blob2_z'].values[0])),
        blob_energies=(df_sig['eblob1'].values[0], df_sig['eblob2'].values[0]),
        text_pos=((35, 27), (-35, 40)),
    ),
    right_kwargs=dict(
        title='Candidate background event',
        plot_lims=((-480, -50), (900, 1200)),
        blob_locations=((df_bkg['blob1_x'].values[0], df_bkg['blob1_z'].values[0]),
                        (df_bkg['blob2_x'].values[0], df_bkg['blob2_z'].values[0])),
        blob_energies=(df_bkg['eblob1'].values[0], df_bkg['eblob2'].values[0]),
        text_pos=((0, 40), (-42, -55)),
        y_axis_label=False,
    ),
    savepath=f'plots/signal_vs_background_{sig_evt}_{bkg_evt}',
)


# ---- event browser (was commented out) -------------------------------
# files = glob.glob('data/*.h5')
# for f in files[::-1]:
#     try:
#         q = pd.read_hdf(f, 'RECO/Events')
#         for evt, df in q.groupby('event'):
#             if (df.Ec.sum() > 1.4) & (df.Ec.sum() < 1.7):
#                 if df.npeak.nunique() == 1:
#                     print(f"file {f.split('/')[-1:]} evt {evt}")
#                     plot_event_single(df, title=f'{evt}', savepath=None)
#     except Exception as e:
#         print(e)
