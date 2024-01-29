import numpy as np


def shift_to_bin_centers(x):
    """
    Return bin centers, given bin lower edges.
    """
    return x[:-1] + np.diff(x) * 0.5

# this only exists to test what this function outputs from beersheba.py

def main():
    sample_width = [15.55, 15.55]
    data = (np.array([-69.975]), np.array([396.525]))
    weight = [0.02399314]

    # grid over which we interpolate
    det_grid = [np.linspace(234.5, -234.5, endpoint = True, num = 470), np.linspace(234.5, -234.5, endpoint = True, num = 470)]
    print(det_grid)



    ranges = [[coord.min() - 1.5 * sw, coord.max() + 1.5 * sw] for coord, sw in zip(data, sample_width)]
    allbins   = [np.arange(rang[0], rang[1] + np.finfo(np.float32).eps, sw) for rang, sw in zip(ranges, sample_width)]
    Hs, edges = np.histogramdd(data, bins=allbins, normed=False, weights=weight)

    inter_points = np.meshgrid(*(shift_to_bin_centers(edge) for edge in edges), indexing='ij')
    inter_points = tuple      (inter_p.flatten() for inter_p in inter_points)


    print(Hs)
    print(inter_points)


    #Hs, inter_points = interpolate_signal(Hs, inter_points, ranges, det_grid, inter_method)




main()