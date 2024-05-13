# Script that collates all the efficiency data into one large h5 file for processing.

PORT=$1

# source IC for python packages
source /gluster/data/next/software/IC_sophronia/config_ic.sh

python3 /gluster/home/jwaiton/scripts/bin/FOM_producer.py /gluster/data/next/files/TOPOLOGY_John/HYPPOS_DATA_QTHR/BEERSHEBA_STUDY/ITER_ECUT/ ${PORT} fiducial_1_${PORT} ../fiducial_1_${PORT}.h5 

