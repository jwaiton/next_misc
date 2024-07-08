
## Configure scisoft software products
source /gluster/data/next/software/IC_satkill/config_ic.sh
echo "IC sourced"


echo "Producing data"
python3 /gluster/home/jwaiton/scripts/bin/collect_signal_events.py
