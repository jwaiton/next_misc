INPUT_D=$1
OUTPUT_D=$2
PORT=$3


## Configure scisoft software products
source /gluster/data/next/software/IC_satkill/config_ic.sh
echo "IC sourced"

echo "Producing signal information"
python3 /gluster/home/jwaiton/scripts/bin/cluster_signal_collect.py $INPUT_D $OUTPUT_D $PORT
