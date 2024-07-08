PORT=$1

## Configure scisoft software products
source /gluster/data/next/software/IC_sophronia/config_ic.sh
echo "IC sourced"

DIRECTORY="/gluster/data/next/files/TOPOLOGY_John/HYPPOS_DATA_QTHR/BEERSHEBA_STUDY/blobR_voxelS/NEW_POSITRON_TEST/"

echo "Producing FOM"
python3 /gluster/home/jwaiton/scripts/bin/cluster_FOM_calc.py $DIRECTORY 
