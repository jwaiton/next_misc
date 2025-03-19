import sys,os,os.path
sys.path.append("../../") # if you move files around, you need to adjust this!
sys.path.append(os.path.expanduser('~/code/eol_hsrl_python'))
sys.path.append("/home/e78368jw/Documents/NEXT_CODE/next_misc/")
os.environ['ICTDIR']='/home/e78368jw/Documents/NEXT_CODE/IC'

import numpy  as np
import pandas as pd

from typing  import List
from typing  import Tuple
from typing  import Callable
from typing  import Optional

from scipy                  import interpolate
from scipy.signal           import fftconvolve
from scipy.signal           import convolve
from scipy.spatial.distance import cdist
from scipy                  import ndimage as ndi

from IC.invisible_cities.core .core_functions import shift_to_bin_centers
from IC.invisible_cities.core .core_functions import in_range

from IC.invisible_cities.types.symbols       import InterpolationMethod
from IC.invisible_cities.types.symbols       import CutType



def generate_satellite_mask(im_deconv, satellite_max_size, e_cut, cut_type):
    '''
    An adaptation to the scikit-image (v0.24.0) function [1], identifies 
    satellite energy depositions within deconvolution image by size
    and proximity to other depositions.

    In practice, input array is a set of boolean 'pixels' that it then categorises as 
    satellite or non-satellite bundles based on given parameters.

    Returns the mask required to remove satellites as done in `richardson_lucy()`
    
    Parameters
    ----------
    im_deconv                 : Deconvoluted 2D array
    satellite_max_size        : Maximum size of satellite deposit, above which they are considered 'real'.
    e_cut                     : Cut over the deconvolution output, arbitrary units or percentage
    cut_type                  : Cut mode to the deconvolution output (`abs` or `rel`) using e_cut
                                `abs`: cut on the absolute value of the hits.
                                `rel`: cut on the relative value (to the max) of the hits.

    Returns
    ----------
    array       : boolean mask of all labelled satellite deposits

    References
    ----------
    .. [1] https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/misc.py#L59-L151
    
    '''

    # apply mask to copy
    im_mask = im_deconv.copy()

    if cut_type is CutType.abs:
        vis_mask = im_mask
    elif cut_type is CutType.rel:
        vis_mask = im_mask / im_mask.max()
    
    im_mask = np.where(vis_mask < e_cut, 0, 1)

    # label deposits within the array
    # hardcoded to include diagonals in the grouping stage (2)
    footprint = ndi.generate_binary_structure(im_mask.ndim, 2)
    ccs, _ = ndi.label(im_mask, footprint)
    # count the bins of each labelled deposit
    component_sizes = np.bincount(ccs.ravel())
    # check if no satellites within deposit
    if len(component_sizes) == 2:
        # Return a fully False array, so that no objects get removed
        return np.full(im_deconv.shape, False)

    
    # create boolean array for each label of satellite & non-satellite
    too_small = component_sizes < satellite_max_size
    # apply boolean array to labelled array to hold satellites as True
    too_small_mask = too_small[ccs]
    # return mask to remove satellites
    return too_small_mask


def test_satkill(max_size):
    data = np.load('/home/e78368jw/Documents/NEXT_CODE/IC/invisible_cities/database/test_data/satellite_array.npy')
    print(f"Initial array:\n{data}")

    # test zero satellite
    mask = generate_satellite_mask(data, max_size, 0.5, CutType.rel)
    data[mask] = 0
    print(f"Array with sat_max_size {max_size}:\n{data}")




def main():
    
    max_size_small = 0
    max_size_large = 9999

    data = np.load('/home/e78368jw/Documents/NEXT_CODE/IC/invisible_cities/database/test_data/satellite_array.npy')
    print(f"Initial array:\n{data}")

    # test zero satellite
    mask = generate_satellite_mask(data, max_size_small, 0.5, CutType.rel)
    data[mask] = 0
    print(f"Array with sat_max_size 0:\n{data}")

    mask = generate_satellite_mask(data, max_size_large, 0.5, CutType.rel)
    data[mask] = 0
    print(f"Array with sat_max_size 9999:\n{data}")

    test_satkill(-1)

    test_satkill('i')





main()
