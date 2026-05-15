import glob
from brokenaxes import brokenaxes
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    with tb.open_file('data/S1_S2_plot/run_15281_0001_ldc1_trg2.waveforms.h5', "r") as h5in:
        rwf_data = h5in.root.RD.pmtrwf

        times       = np.arange(0, len(np.sum(rwf_data[0], axis = 0))*25, 25)
        #import pdb; pdb.set_trace()
        rebin_times, rebin_widths, rebin_wf =  rebin_times_and_waveforms(times, widths = np.tile(25, (len(times), 1)), waveforms = np.array([np.sum(rwf_data[2], axis = 0)]), rebin_stride = 160)
        #pdb.set_trace()
        summed_data = np.sum(rebin_wf, axis = 0)
        fig = plt.figure()
        normalised_baselined_flipped = -summed_data + np.median(summed_data)
        normalised_baselined_flipped = normalised_baselined_flipped / np.max(normalised_baselined_flipped)
        bax = brokenaxes(ylims=((0.7e6, 0.85e6), (1.3e6, 2.0e6)), hspace = 0.05)

        bax.plot(normalised_baselined_flipped, rebin_times )
        for ax in bax.axs:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.2f}'))
        plt.show()


    # plot both together
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(1, 2, width_ratios = [1,3], wspace = 0.05)

    ax_left = fig.add_subplot(gs[1])
    pitch = 15.55
    xx = np.arange(data.X.min(), data.X.max() + pitch, pitch)
    zz = np.sort(data.Z.unique())
    h = ax_left.hist2d(data.X, data.Z, bins=[xx, zz], weights=data.Q, cmin=0.0001);
    ax_left.set_xlim(ax_left.get_xlim()[0] - 10, ax_left.get_xlim()[1] + 10)
    ax_left.set_ylim(ax_left.get_ylim()[0] - 35, ax_left.get_ylim()[1] + 20)
    ax_left.set_xlabel('X (mm)')
    ax_left.set_ylabel('Z (mm)')

    ax_left.yaxis.set_label_position('right')
    ax_left.yaxis.tick_right()
    ax_left.spines['left'].set_visible(True)
    ax_left.spines['right'].set_visible(True)

    # right subplot - brokenaxes

    bax = brokenaxes(ylims=((0.79e6, 0.81e6), (1.38e6, 1.76e6)), hspace=0.0, d=0.005, subplot_spec=gs[0])
    bax.plot(normalised_baselined_flipped, rebin_times)
    bax.axhspan(0.799e6, 0.806e6, alpha = 0.3, color = 'orange', label = 'S1')
    bax.axhspan(1.38e6, 1.74e6, alpha = 0.2, color = 'blue', label = 'S2')
    handles, labels = [], []
    for ax in bax.axs:
        h, l = ax.get_legend_handles_labels()
        handles = h
        labels = l
    bax.set_ylabel('time (ms)', labelpad=50)

    ax_left.legend(handles, labels, loc = 'upper left')

    for ax in bax.axs:
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
    for ax in bax.axs:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.2f}'))
    plt.savefig('plots/S1S2_plots/S1S2_visual.pdf')

    plt.savefig('plots/S1S2_plots/S1S2_visual.png')
    plt.show()


main()
