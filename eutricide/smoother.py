import sys,os,os.path
sys.path.append("../../")   # cite IC from parent directory
                            # NOTE if you can't import IC stuff, its because of the
                            # above line
#sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
os.environ['ICTDIR']='/home/e78368jw/Documents/NEXT_CODE/IC'

import os
import glob
import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import cv2 as cv


from IC.invisible_cities.reco.psf_functions    import create_psf
from IC.invisible_cities.reco.psf_functions    import hdst_psf_processing
from IC.invisible_cities.reco.psf_functions    import add_empty_sensors_and_normalize_q
from IC.invisible_cities.reco.psf_functions    import add_variable_weighted_mean

import IC.invisible_cities.core.core_functions as     coref
import IC.invisible_cities.io  .dst_io         as     dstio

from IC.invisible_cities.database              import load_db
from IC.invisible_cities.io.dst_io import df_writer

import functions as func


def smooth(psf, kernel_factor = 5):
    #generate kernel
    av_factor = kernel_factor
    kernel    = np.ones((av_factor, av_factor),np.float32)/(av_factor**2)

    finished_psf = []

    # apply across each z slice
    for z, psf_z in psf.groupby('z'):
        # reshape
        nx     = psf_z.xr.unique().size
        ny     = psf_z.yr.unique().size
        matrix = psf_z.factor.values.reshape(nx, ny)
        # smooth
        dst    = cv.filter2D(matrix,-1,kernel)

        # abusing symmetry
        xyrange = (psf_z.xr.min(), psf.xr.max())
        xr_vals = np.linspace(xyrange[0], xyrange[1], nx)
        yr_vals = np.linspace(xyrange[0], xyrange[1], ny)

        xr, yr = np.meshgrid(xr_vals, yr_vals, indexing='ij')

        # make a psf dataframe
        psf_smooth = pd.DataFrame({
                                    'xr'     : xr.flatten(),
                                    'yr'     : yr.flatten(),
                                    'zr'     : 0.0,
                                    'x'      : 0.0,
                                    'y'      : 0.0,
                                    'factor' : dst.flatten(),
                                    'z'      : z,
                                    'nevt'   : psf_z['nevt'] # this should map normally?
        })
        finished_psf.append(psf_smooth)

    return pd.concat(finished_psf)

def smooth_and_write(input_path, output_path):


    # load dst and extract bin size
    psf_data    = dstio.load_dst(f'{input_path}')
    bin_size_xy = psf_data.xr.diff().unique()[-1]
    # smooth
    smooth_data = smooth(psf_data)

    with tb.open_file(output_path, 'w') as h5out:
        df_writer(h5out, smooth_data,
                  'PSF', 'PSFs',
                  descriptive_string = f"PSF with {bin_size_xy} mm bin size")


input_path = sys.argv[1]
print(f'Input: {input_path}')
output_path = sys.argv[2]
print(f'Output: {output_path}')
rebin_value = sys.argv[3]

smooth_and_write(input_path, output_path)
