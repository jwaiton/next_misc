import os
import glob
import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import re


###################################################
#
#                  PSF RELATED FUNCTIONS
#
###################################################


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


def plot_psf(df, label, norm = False):

    if norm:
        plt.plot((np.flip(-df.rr)).append(df.rr), (np.flip(normalise(df.factor))).append(normalise(df.factor)), label = label)
    else:
        plt.plot((np.flip(-df.rr)).append(df.rr), (np.flip(df.factor)).append(df.factor), label = label)


def z_specific_rr(df, z_val):
    '''
    Write script here that implements above rr function but across specific z value from the PSF.
    '''
    dfz = df[df.z == z_val]
    return rr_process(dfz)


def create_plots_psf(df, z, label = 'None', norm = False):
    q = z_specific_rr(df, z)
    plot_psf(q, label, norm)

    return q


# human sorting
def tryint(s):
    """
    Return an int if possible, or `s` unchanged.
    """
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """
    Turn a string into a list of string and number chunks.

    >>> alphanum_key("z23a")
    ["z", 23, "a"]

    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def human_sort(l):
    """
    Sort a list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
