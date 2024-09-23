import os
import sys
import random
import tables as tb
import numpy  as np
import pandas as pd
import scipy.spatial.distance as scidist
import invisible_cities.core.fit_functions  as fitf
from   invisible_cities.core.stat_functions import poisson_sigma
import invisible_cities.reco.dst_functions  as dstf
from   invisible_cities.core.core_functions import shift_to_bin_centers
from invisible_cities.core.testing_utils import assert_hit_equality

from  skimage.morphology import skeletonize_3d

from invisible_cities.evm                  import event_model as evm
from invisible_cities.evm  .event_model    import Cluster, Hit
from invisible_cities.types.ic_types       import xy
from invisible_cities.reco import paolina_functions    as plf


def find_skel_extr_voxels(track_graph, skel_extr):

    voxels = np.array(list(track_graph.nodes()))
    v_pos  = np.array([v.XYZ for v in voxels])
    dist_extr_v = scidist.cdist(np.array([skel_extr]), np.array(v_pos))
    distances   = np.array(dist_extr_v[0])
    
    v = voxels[np.argwhere(distances == distances.min())]
    
    return v[0][0]


def hits_in_blob(track_graph,
                 radius,
                 blob_pos,
                 extreme):
    """Returns the hits that belong to a blob."""
    distances         = plf.shortest_paths(track_graph)
    dist_from_extreme = distances[extreme]
    diag              = np.linalg.norm(extreme.size)
    print(blob_pos)
    blob_hits = []
    # First, consider only voxels at a certain distance from the end-point, along the track.
    # We allow for 1 extra contiguity, because this distance is calculated between
    # the centres of the voxels, and not the hits. In the second step we will refine the
    # selection, using the euclidean distance between the blob position and the hits.
    for v in track_graph.nodes():
        voxel_distance = dist_from_extreme[v]
        if voxel_distance < radius + diag:
            for h in v.hits:
                hit_distance = np.linalg.norm(blob_pos - h.pos)
                if hit_distance < radius:
                    blob_hits.append(h)

    return blob_hits


def blob_energies_hits(track_graph, radius, blob_pos_a, blob_pos_b):
    """Return the energies and the hits of the blobs.
       For each pair of observables, the one of the blob of largest energy is returned first."""
   # distances = plf.shortest_paths(track_graph)
   # a, b, _   = plf.find_extrema_and_length(distances)
    extr_a = find_skel_extr_voxels(track_graph, blob_pos_a)
    extr_b = find_skel_extr_voxels(track_graph, blob_pos_b)
    
    ha = hits_in_blob(track_graph, radius, blob_pos_a, extr_a)
    hb = hits_in_blob(track_graph, radius, blob_pos_b, extr_b)

    voxels = list(track_graph.nodes())
    e_type = voxels[0].Etype
    Ea = sum(getattr(h, e_type) for h in ha)
    Eb = sum(getattr(h, e_type) for h in hb)

    # Consider the case where voxels are built without associated hits
    #if len(ha) == 0 and len(hb) == 0 :
    #    Ea = plf.energy_of_voxels_within_radius(distances[a], radius)
   #     Eb = plf.energy_of_voxels_within_radius(distances[b], radius)

    if Eb > Ea:
        return (Eb, Ea, hb, ha, blob_pos_b, blob_pos_a)
    else:
        return (Ea, Eb, ha, hb, blob_pos_a, blob_pos_b)



def apply_skeleton(df):
    
    x = df.X.values
    y = df.Y.values
    z = df.Z.values
    
    xx = np.linspace(x.min() - 0.5, x.max() + 0.5, int(x.max() - x.min() + 2))
    yy = np.linspace(y.min() - 0.5, y.max() + 0.5, int(y.max() - y.min() + 2))
    zz = shift_to_bin_centers(df.Z.unique())
    
    zz = np.append(zz, zz[-1] + 1)
    zz = np.append(zz[0] - 1, zz)
    values = np.histogramdd((x, y, z), bins=[xx, yy, zz])    
    val    = values[0].transpose(2,0,1).flatten()
    digitize = np.where(values[0] > 0, 1, 0)
    skeleton = skeletonize_3d(digitize)
    #if skeleton.sum() > 0:
    skeleton_mask = np.where(skeleton == 1)
    x_skel = shift_to_bin_centers(xx)[skeleton_mask[0]]
    y_skel = shift_to_bin_centers(yy)[skeleton_mask[1]]
    z_skel = shift_to_bin_centers(zz)[skeleton_mask[2]]
    e_skel = values[0][skeleton_mask]
    
    return pd.DataFrame({'X':x_skel, 'Y':y_skel, 'Z':z_skel, 'E':e_skel})



def convert_df_to_hits(df):
    return [Hit(0, Cluster(0, xy(h.X,h.Y), xy(0,0), 0), h.Z, h.E, xy(0, 0))
            for h in df.itertuples(index=False)]


def get_skel_extremes(vox_size         ,
                      energy_type      ,
                      strict_vox_size  ,
                      hitc):
        
    voxels     = plf.voxelize_hits(hitc.hits, vox_size, strict_vox_size, energy_type)
    tracks     = plf.make_track_graphs(voxels)

    df = pd.DataFrame(columns=['event', 'trackID', 'energy', 
                               'skel_extr1_x', 'skel_extr1_y', 'skel_extr1_z',
                               'skel_extr2_x', 'skel_extr2_y', 'skel_extr2_z'])

    if (len(voxels) == 0):
        return df

    vox_size_x = voxels[0].size[0]
    vox_size_y = voxels[0].size[1]
    vox_size_z = voxels[0].size[2]

    def get_track_energy(track):
        return sum([vox.Ehits for vox in track.nodes()])
    #sort tracks in energy
    tracks     = sorted(tracks, key = get_track_energy, reverse = True)

    track_hits = []

    for c, t in enumerate(tracks, 0):
        tID = c
        energy = get_track_energy(t)

        extr1, extr2 = plf.find_extrema(t)
        extr1_pos = extr1.XYZ
        extr2_pos = extr2.XYZ

#        blob1_pos = plf.blob_centre(extr1)        
#        blob2_pos = plf.blob_centre(extr2)
        blob1_pos = extr1_pos
        blob2_pos = extr2_pos

        list_of_vars = [hitc.event, tID, energy, 
                        blob1_pos[0], blob1_pos[1], blob1_pos[2],
                        blob2_pos[0], blob2_pos[1], blob2_pos[2]]

        df.loc[c] = list_of_vars
        try:
            types_dict
        except NameError:
            types_dict = dict(zip(df.columns, [type(x) for x in list_of_vars]))
            
        #change dtype of columns to match type of variables
        df = df.apply(lambda x : x.astype(types_dict[x.name]))

        return df



def create_tracks_with_skel_extremes(vox_size       ,
                                     energy_type      ,
                                     strict_vox_size  ,
                                     blob_radius,
                                     df_extr,
                                     hitc):
    voxels = plf.voxelize_hits(hitc.hits, vox_size, strict_vox_size, energy_type)
    tracks = plf.make_track_graphs(voxels)

    df = pd.DataFrame(columns=['event', 'trackID', 'energy', 'length',
                               'numb_of_voxels', 'numb_of_hits',
                               'numb_of_tracks', 'x_min', 'y_min', 'z_min',
                               'x_max', 'y_max', 'z_max', 'r_max',
                               'x_ave', 'y_ave', 'z_ave',
                                'extreme1_x', 'extreme1_y', 'extreme1_z',
                                'extreme2_x', 'extreme2_y', 'extreme2_z',
                                'blob1_x', 'blob1_y', 'blob1_z',
                                'blob2_x', 'blob2_y', 'blob2_z',
                                'eblob1', 'eblob2', 'ovlp_blob_energy',
                                'vox_size_x', 'vox_size_y', 'vox_size_z'])

    if (len(voxels) == 0):
        return df

    vox_size_x = voxels[0].size[0]
    vox_size_y = voxels[0].size[1]
    vox_size_z = voxels[0].size[2]
    
    def get_track_energy(track):
            return sum([vox.Ehits for vox in track.nodes()])
        #sort tracks in energy
    tracks     = sorted(tracks, key = get_track_energy, reverse = True)

    track_hits = []
 
    for c, t in enumerate(tracks, 0):
        tID = c
        energy = get_track_energy(t)
        length = plf.length(t)
        numb_of_hits = len([h for vox in t.nodes() for h in vox.hits])
        numb_of_voxels = len(t.nodes())
        numb_of_tracks = len(tracks   )
 
        min_x = min([h.X for v in t.nodes() for h in v.hits])
        max_x = max([h.X for v in t.nodes() for h in v.hits])
        min_y = min([h.Y for v in t.nodes() for h in v.hits])
        max_y = max([h.Y for v in t.nodes() for h in v.hits])
        min_z = min([h.Z for v in t.nodes() for h in v.hits])
        max_z = max([h.Z for v in t.nodes() for h in v.hits])
        max_r = max([np.sqrt(h.X*h.X + h.Y*h.Y) for v in t.nodes() for h in v.hits])
 
        pos = [h.pos for v in t.nodes() for h in v.hits]
        e   = [getattr(h, energy_type.value) for v in t.nodes() for h in v.hits]
        ave_pos = np.average(pos, weights=e, axis=0)
        
        # classic paolina extremes
        extr1, extr2 = plf.find_extrema(t)
        extr1_pos = extr1.XYZ
        extr2_pos = extr2.XYZ
        
        t_extr = df_extr[df_extr.trackID == tID]
        if len(t_extr) == 0:
            blob_pos1 = np.array([1.e6, 1.e6, 1.e6])
            blob_pos2 = np.array([1.e6, 1.e6, 1.e6])
            eblob1 = -1
            eblob2 = -1
            overlap = -1
        else:
            blob_pos1 = np.array([t_extr.skel_extr1_x.values[0], t_extr.skel_extr1_y.values[0], t_extr.skel_extr1_z.values[0]])
            blob_pos2 = np.array([t_extr.skel_extr2_x.values[0], t_extr.skel_extr2_y.values[0], t_extr.skel_extr2_z.values[0]])
            e_blob1, e_blob2, hits_blob1, hits_blob2, blob_pos1, blob_pos2  = blob_energies_hits(t, blob_radius, blob_pos1, blob_pos2)
            overlap = sum([h.E for h in set(hits_blob1).intersection(set(hits_blob2))])

        list_of_vars = [hitc.event, tID, energy, length, numb_of_voxels,
                            numb_of_hits, numb_of_tracks,
                            min_x, min_y, min_z, max_x, max_y, max_z, max_r,
                            ave_pos[0], ave_pos[1], ave_pos[2],
                            extr1_pos[0], extr1_pos[1], extr1_pos[2],
                            extr2_pos[0], extr2_pos[1], extr2_pos[2],
                            blob_pos1[0], blob_pos1[1], blob_pos1[2],
                            blob_pos2[0], blob_pos2[1], blob_pos2[2],
                            e_blob1, e_blob2, overlap,
                            vox_size_x, vox_size_y, vox_size_z]
        df.loc[c] = list_of_vars
        try:
            types_dict
        except NameError:
            types_dict = dict(zip(df.columns, [type(x) for x in list_of_vars]))
            
    #change dtype of columns to match type of variables
    df = df.apply(lambda x : x.astype(types_dict[x.name]))
    
    return df

start = int(sys.argv[1])
numb  = int(sys.argv[2])
prod  = str(sys.argv[3])

ifolder = '/scratch/paola/SimMC/beershe/'
#ifolder = '/Users/halmamol/NEXT/files/NEXT100/beersheba/0nubb/'
#ofolder = '/Users/halmamol/NEXT/files/NEXT100/beersheba/0nubb/'
ofolder = '/scratch/halmazan/ofiles/'

base = 'next100.{}'.format(prod)

voxel = 5
blob_radius = 15
vox_size = [voxel, voxel, voxel]

for i in range(start, start+numb):
    ifile = ifolder + 'hits/'   + '{0}.{1}.deconv.h5'.format(base, i)
    ofile = ofolder + 'tracks/' + '{0}.{1}.skel_tracks.R{2}mm.h5'.format(base, i, int(blob_radius))

    try:
        df = pd.read_hdf(ifile, 'DECO/Events')
    except:
        print(f'File {ifile} not good.')
        continue

    print(f'Analyzing file {ifile}')    

    evts   = df.event.unique()
    tracks = []

    for i, evt in enumerate(evts):
        df_evt      = df[df.event == evt]
        skel_df     = apply_skeleton(df_evt)

        skel_hits      = evm.HitCollection(evt, 0)
        skel_hits.hits = convert_df_to_hits(skel_df)
        if len(skel_hits.hits) == 0:
            print(f'Event {evt} has no reconstructed skeleton hits.')
            continue

        skel_extr = get_skel_extremes(vox_size,
                                      energy_type = evm.HitEnergy.E,
                                      strict_vox_size = False,
                                      hitc = skel_hits)

        hitcol      = evm.HitCollection(evt, 0)
        hitcol.hits = convert_df_to_hits(df_evt)
        if len(hitcol.hits) == 0:
            print(f'Event {evt} has no reconstructed hits.')
            continue


        final_df = create_tracks_with_skel_extremes(vox_size,
                                                    evm.HitEnergy.E,
                                                    False,
                                                    blob_radius,
                                                    skel_extr,
                                                    hitcol)

        if len(final_df) == 0: continue

        tracks.append(final_df)

    df_all = pd.concat(tracks)
    df_all.to_hdf(ofile, key='Tracks', mode='w')
