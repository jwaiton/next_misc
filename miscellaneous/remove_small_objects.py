def remove_small_objects(data, min_size = 4, connectivity = 2):
    '''
    An adapted function from scikit-image
    https://github.com/scikit-image/scikit-image/blob/main/skimage/morphology/misc.py#L59-L151
    Pulled in here because of package mismatches (scikit-image breaks my conda env, this is easier I think)
    set min_size to 4-ish, as per-iteration you wouldn't expect more than 4 pixels (interpolated) to be formed by a satellite
    connectivity set to 2 to catch diagonals
    '''

    array = data.copy().astype(bool)
    # Not connections if satellite size is zero
    if min_size == 0:
        return array


    # label the blobs within the array
    footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
    ccs = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, footprint, output=ccs)
    
    # count the bins of each labelled blob
    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported.")

    # check if no-satellites
    if len(component_sizes) == 2:
        return array
    
    too_small = component_sizes < min_size
    too_smal_maks = too_small[ccs]
    array[too_small_mask] = 0

    return array



