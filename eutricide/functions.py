import os
import glob
import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt


def plot_z_bins(psf, bins, title, show = True):
    plt.hist(psf.z, bins = bins)
    plt.title(title)
    plt.xlabel('z (mm)')
    if show:
        plt.show()


def normalise(data):
    return (data)/(max(data))


def rr_process(df_in):    
    return (df_in.assign(rr = (df_in.xr**2 + df_in.yr**2)**0.5)
              .groupby("rr")
              .agg({'factor' : "mean"})
              .reset_index()
           )


def plot_psf(df, label, norm = False, color = None):


    if norm:
        plt.plot(np.concatenate([np.flip(-df.rr.values), df.rr.values]), 
             np.concatenate([np.flip(normalise(df.factor.values)), normalise(df.factor.values)]), 
             label=label, 
             color=color if color is not None else None)
        #plt.plot(np.concatenate([np.flip(-df.rr.values),df.rr.values]), (np.concatentate(np.flip(normalise(df.factor.values))), (normalise(df.factor.values))), label = label)
    else:
        plt.plot((np.flip(-df.rr)).append(df.rr), (np.flip(df.factor)).append(df.factor),
                  label = label,
                  color=color if color is not None else None)


def z_specific_rr(df, z_val):
    '''
    Write script here that implements above rr function but across specific z value from the PSF.
    '''
    dfz = df[df.z == z_val]
    return rr_process(dfz)


def create_plots_psf(df, z, label = 'None', norm = False, color = None):
    q = z_specific_rr(df, z)
    plot_psf(q, label, norm, color)

    return q
