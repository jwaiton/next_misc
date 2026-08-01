import glob
from brokenaxes import brokenaxes
import plot_info
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import tables as tb
import sys, os

sys.path.append('/home/e78368jw/Documents/NEXT_CODE/IC/')
os.environ['ICTDIR']='/home/e78368jw/Documents/NEXT_CODE/IC/'

from invisible_cities.cities.components   import track_blob_info_creator_extractor
from invisible_cities.reco.peak_functions import rebin_times_and_waveforms
from invisible_cities.reco.deconv_functions import drop_isolated_clusters
from  invisible_cities.evm.event_model        import Cluster, Hit
from invisible_cities.reco.paolina_functions import voxelize_hits
from invisible_cities.types.ic_types import xy


def _setup_3d_panel(fig, position, view_elev=-25, view_azim=50,
                     xlim=None, ylim=None, zlim=None):
    """
    Create and style a single 3D axes panel with consistent
    labels, hidden tick labels, view angle, and (optionally)
    shared data limits across panels.

    position : subplot spec, e.g. (nrows, ncols, index) or a
               3-digit int like 111, or a SubplotSpec from GridSpec.
    """
    ax = fig.add_subplot(*position, projection='3d') if isinstance(position, tuple) \
         else fig.add_subplot(position, projection='3d')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    ax.view_init(view_elev, view_azim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_zlabel('Z (mm)', labelpad=-10)  # push it down/away from the title
    ax.set_xlabel('X (mm)', labelpad=-10)
    ax.set_ylabel('Y (mm)', labelpad=-10)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if zlim is not None: ax.set_zlim(zlim)

    return ax

from matplotlib.colors import Normalize
from matplotlib import cm


def plot_hits_3d(ax, df, evt, norm, alpha=0.90, min_s=10, max_s=15):
    """
    Draw a scatter-style 3D hit display onto an existing axes `ax`,
    using a shared Normalize instance so color scale matches other
    panels (e.g. a paired voxel plot).
    """
    sub = df[df.event == evt]
    xt, yt, zt, et = sub.X, sub.Y, sub.Z, sub.E
    ets = et > 0
    max_val = max(et[ets])
    scaled_clipped = [max((v / max_val) * max_s, min_s) for v in et[ets]]
    p = ax.scatter(xt[ets], yt[ets], zt[ets], c=et[ets],
                    alpha=alpha, cmap='viridis', s=scaled_clipped)
    return p

def plot_voxels_3d(ax, df, base_vsize=12):
    """
    Draw a voxelised 3D event display onto an existing axes `ax`.
    Uses its own locally-computed normalization (0 to max voxel
    energy) rather than a shared scale passed in from the caller.
    """
    voxels = voxelize_hits(df, np.array([base_vsize]*3), False)
    vsizex, vsizey, vsizez = voxels[0].size

    min_corner_x = min(v.X for v in voxels) - vsizex/2.
    min_corner_y = min(v.Y for v in voxels) - vsizey/2.
    min_corner_z = min(v.Z for v in voxels) - vsizez/2.

    x = [np.round(v.X/vsizex) for v in voxels]
    y = [np.round(v.Y/vsizey) for v in voxels]
    z = [np.round(v.Z/vsizez) for v in voxels]
    e = [v.E for v in voxels]

    x_min, y_min, z_min = int(min(x)), int(min(y)), int(min(z))
    x_max, y_max, z_max = int(max(x)), int(max(y)), int(max(z))

    VOXELS = np.zeros((x_max-x_min+1, y_max-y_min+1, z_max-z_min+1))
    cmap = cm.viridis
    norm = Normalize(vmin=0, vmax=max(e))  # local normalization, own scale
    colors = np.empty(VOXELS.shape, dtype=object)

    for q in range(len(z)):
        VOXELS[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = 1
        rgba = list(cmap(norm(e[q])))
        rgba[3] = max(0.8, norm(e[q]))
        colors[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = tuple(rgba)

    a, b, c = np.indices((x_max-x_min+2, y_max-y_min+2, z_max-z_min+2))
    a = a*vsizex + min_corner_x
    b = b*vsizey + min_corner_y
    c = c*vsizez + min_corner_z

    ax.voxels(a, b, c, VOXELS, facecolors=colors)
    return cmap, norm  # returned so caller can build its own colorbar if needed


def plot_hits_and_voxels(df, evt, panel_size=5.0, base_vsize=12,
                          suptitle=None, savepath=None, clrbar=True):
    fig = plt.figure(figsize=(panel_size*2, panel_size))
    if suptitle:
        fig.suptitle(suptitle, y=0.95)

    sub = df[df.event == evt]
    ets = sub.E > 0
    pad = base_vsize
    xlim = (sub.X[ets].min() - pad, sub.X[ets].max() + pad)
    ylim = (sub.Y[ets].min() - pad, sub.Y[ets].max() + pad)
    zlim = (sub.Z[ets].min() - pad, sub.Z[ets].max() + pad)

    # --- shared normalization across both panels ---
    voxels = voxelize_hits(sub, np.array([base_vsize]*3), False)
    voxel_e = [v.E for v in voxels]
    global_max = max(max(sub.E[ets]), max(voxel_e))
    norm = Normalize(vmin=0, vmax=global_max)
    cmap = cm.viridis

    ax1 = _setup_3d_panel(fig, (1, 2, 1), xlim=xlim, ylim=ylim, zlim=zlim)
    plot_hits_3d(ax1, df, evt, norm=norm)
    ax1.set_title('Candidate track')

    ax2 = _setup_3d_panel(fig, (1, 2, 2), xlim=xlim, ylim=ylim, zlim=zlim)
    plot_voxels_3d(ax2, sub, base_vsize=base_vsize)
    ax2.set_title('Voxelised track')

    if clrbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=[ax1, ax2], shrink=0.6, label='Energy (keV)')

    fig.subplots_adjust(wspace=0.05)

    if savepath:
        fig.savefig(savepath + '.png', dpi=300)
        fig.savefig(savepath + '.pdf')

    return fig, (ax1, ax2)


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
    fig.suptitle(f"{evt}")
    plt.show()





def main():
    drop_clusters = drop_isolated_clusters([16., 16., 4.], 3, ['Ec', 'E'])
    data = pd.read_hdf('data/S1_S2_plot/run_15281_0001_ldc1_trg2.v2.3.1.20250429.HEDesman.sophronia.h5', 'RECO/Events')
    data = data[data.event == 842]
    data = drop_clusters(data)
    print(data)

    for evt, df in data.groupby('event'):
        raw_plotter(df, evt)

        plot_info.apply_style(scale_factor = (1/1.35))
        plot_hits_and_voxels(
            df, evt,
            base_vsize=21,
            savepath=f'plots/voxelisation/event_{evt}',
            clrbar = False
        )




main()
