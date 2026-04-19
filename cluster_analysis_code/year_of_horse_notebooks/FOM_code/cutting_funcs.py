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
