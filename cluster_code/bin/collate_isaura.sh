# Script that collates all the efficiency data into one large h5 file for processing.

PORT=$1

# source IC for python packages
source /gluster/data/next/software/IC_sophronia/config_ic.sh
export HDF5_USE_FILE_LOCKING='FALSE'
python3 /gluster/home/jwaiton/scripts/bin/FOM_producer.py /gluster/data/next/files/TOPOLOGY_John/HYPPOS_DATA_QTHR/BEERSHEBA_STUDY/blobR_voxelS/ ${PORT} fiducial_isaura_${PORT} ../fiducial_isaura_${PORT}.h5 

