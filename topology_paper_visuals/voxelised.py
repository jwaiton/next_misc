import glob
from brokenaxes import brokenaxes
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
from invisible_cities.io.hits_io          import hits_from_df
from invisible_cities.reco.peak_functions import rebin_times_and_waveforms
from invisible_cities.reco.hits_functions  import drop_isolated_clusters
from  invisible_cities.evm.event_model        import Cluster, Hit
from invisible_cities.reco.paolina_functions import voxelize_hits
from invisible_cities.types.ic_types import xy

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


def plotter_3d(df, evt, cut_n_drop = True, show = True, clrbar = True, alpha = 0.90, min_s = 10, max_s = 15, cut_sensors = None, drop_sensors = None):
    '''
    evt_interest - df
    evt          - event number

    '''
    # plot
    evt_interest = df[df.event == evt]


    xt = df.X
    yt = df.Y
    zt = df.Z
    et = df.E

    fig = plt.figure(figsize=(12,8))
    #fig.suptitle('3D post deconvolution ' + str(evt), fontsize=30)
    fig.suptitle(f'Candidate track', fontsize=36, y = 0.92)
    ax = fig.add_subplot(111, projection='3d')



    ets = et > 0 # eliminate small things for measurement

    max_val = max(et[ets])
    scaled_clipped = [max((v / max_val) * max_s, min_s) for v in et[ets]]

    #p = ax.scatter(x[em], y[em], z[em], c=e[em], alpha=0.3, cmap='viridis')
    #plt_sphere([(-track.blob2_x.values[0], -track.blob2_y.values[0], -track.blob2_z.values[0])], [blobR])
    p = ax.scatter([xt[ets]], yt[ets], zt[ets], c=et[ets], alpha=alpha, cmap='viridis', s = scaled_clipped)#, s = et[ets])
    #q = ax.scatter(xt, yt, zt, alpha = 0.3, color = 'red')

    # overlay the blobs and their radii
    #if clrbar:
    #    cb = fig.colorbar(p, ax=ax)
    #    cb.set_label('Energy (keV)')



    ax.set_xlabel('x (mm)')#, labelpad = 15)#,fontsize=16)
    ax.set_ylabel('y (mm)')#, labelpad = 15)#,fontsize=16)
    ax.set_zlabel('z (mm)')#, labelpad = 20)#,fontsize=16)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    ax.view_init(-25, 50)

    #ax.set_xlim([-300, -100])
    #ax.set_ylim([250, 450])
    #ax.set_zlim([1600, 1800])
    #ax.view_init(20, -150)

    #plt.savefig(f'gif_making/deconv/angle_{i}.png')
    #plt.savefig(f'plots/hits_3d_{evt}.pdf')
    if show:
        plt.savefig('plots/voxelisation/hit_track.png', pad_inches=0.5)
        plt.savefig('plots/voxelisation/hit_track.pdf')
        plt.show()




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

    for q in range(0, len(z)):
        VOXELS[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = 1
        rgba = list(cmap(norm(e[q])))
        rgba[3] = max(0.8, norm(e[q]))  # minimum alpha of 0.1
        colors[int(x[q])-x_min][int(y[q])-y_min][int(z[q])-z_min] = tuple(rgba)


    # and plot everything
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    #a,b,c is spacing in mm needs an extra dim
    a,b,c = np.indices((x_max-x_min+2, y_max-y_min+2, z_max-z_min+2))
    a = a*vsizex + min_corner_x
    b = b*vsizey + min_corner_y
    c = c*vsizez + min_corner_z

    # a, b, c are the corners of the voxels
    ax.voxels(a, b, c, VOXELS, facecolors=colors)

    ax.set_xlabel('x (mm)')#, labelpad = 15)#,fontsize=16)
    ax.set_ylabel('y (mm)')#, labelpad = 15)#,fontsize=16)
    ax.set_zlabel('z (mm)')#, labelpad = 20)#,fontsize=16)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])



    ax.view_init(-25, 50)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #cb = fig.colorbar(sm, ax=ax, pad = 0.1)
    #cb.set_label('Energy (keV)')

    fig.suptitle('Voxelised track', y = 0.92, fontsize = '36')
    #ax.view_init(-160, 90)

    plt.savefig('plots/voxelisation/voxelisation.png', pad_inches=0.5)
    plt.savefig('plots/voxelisation/voxelisation.pdf', pad_inches=0.5)
    plt.show()



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
        plotter_3d(df, evt)
        plot_voxels(df, base_vsize = 21)



main()
