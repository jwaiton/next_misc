import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_hist(df, column = 'energy', binning = 20, title = "Energy plot", output = False, fill = True, label = 'default', x_label = 'energy (MeV)', y_label = 'Counts/bin', range = 0, outliers = None, log = True, data = False, save = False, save_dir = '', alpha = 1):
    '''
    Produce a histogram for a specific column within a dataframe.
    Used for NEXT-100 data analysis. Can be overlaid on top of other histograms.
    
    Args:
        df          :       pandas dataframe
        column      :       column the histogram will plot
        binning     :       number of bins in histogram
        title       :       title of the plot
        output      :       visualise the plot (useful for notebooks)
        fill        :       fill the histogram
        label       :       Add histogram label (for legend)
        x_label     :       x-axis label
        y_label     :       y-axis label
        range       :       range limiter for histogram (min, max)
        outliers    :       Remove outliers by percentile (min, max)
                            If you set a range, this wont work
        log         :       y-axis log boolean
        data        :       output histogram information boolean
        save        :       Save the plot as .png boolean
        save_dir    :       directory to save the plot
        alpha       :       opacity of histogram

    Returns:
        if (data==False):
            None          :       empty return
        if (data==True):
            (cnts,        :       number of counts
            edges,        :       values of the histogram edges
            patches)      :       matplotlib patch object


    '''
    # for simplicity/readability, scoop out the relevant columns from the pandas dataframe.
    energy_vals = df[column].to_numpy()

    
    if outliers is not None:
        lower = np.percentile(energy_vals, outliers[0])
        upper = np.percentile(energy_vals, outliers[1])
        energy_vals  = energy_vals[(energy_vals >= lower) & (energy_vals <= upper)]

    if (range==0):
        range = (np.min(energy_vals), np.max(energy_vals))
    
    # control viewing of hist
    if (fill == True):
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, range = range, alpha = alpha, color='C1')
    else:
        cnts, edges, patches = plt.hist(energy_vals, bins = binning, label = label, histtype='step', linewidth = 2, range = range)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if log == True:
        plt.yscale('log')
    if (save==True):
        if not (save_dir == ''):
            plt.savefig(save_dir + title + ".png")
        else:
            print("Please provide a suitable directory to save the data!")
    if (output==True):
        plt.legend()
        plt.show()
    if (data==True):
        return (cnts, edges, patches)
    else:
        return
