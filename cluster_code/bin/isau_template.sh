## Job options


NUMBER=$(($1+1))
RUNNUMBER=0

INPUT_1=$2   # xy voxel space
INPUT_2=$3   # z voxel space
INPUT_3=$4   # blob radius
INPUT_4=$5   # port value
PORT=$INPUT_4

echo "Applying voxel size: ${INPUT_1} x ${INPUT_1} x ${INPUT_2}"
echo "Applying blob radius: ${INPUT_3}"
echo "Across PORT: ${INPUT_4}"

#####################
# ISAURA PARAMETERS
#####################

VOXEL_SIZE="[${INPUT_1} * mm, ${INPUT_1} * mm, ${INPUT_2} * mm]"
BLOB_RAD="${INPUT_3} * mm"

####################
# FILE LOCATIONS
####################


INFILE="/gluster/data/next/files/TOPOLOGY_John/HYPPOS_DATA_QTHR/BEERSHEBA_STUDY/ITER_ECUT/75_9e-3/PORT_${PORT}/beersheba/beersheba_${NUMBER}_208Tl.h5"
ISAUFILE="/gluster/data/next/files/TOPOLOGY_John/HYPPOS_DATA_QTHR/BEERSHEBA_STUDY/blobR_voxelS/${INPUT_1}_${INPUT_1}_${INPUT_2}/PORT_1a/isaura/isaura_${NUMBER}_208Tl.h5"

# Activating IC

echo "IO files found"
## Configure scisoft software products
source /gluster/data/next/software/IC_sophronia/config_ic.sh
echo "IC sourced"

DIRECTORY="/gluster/data/next/files/TOPOLOGY_John/HYPPOS_DATA_QTHR/BEERSHEBA_STUDY/blobR_voxelS/${INPUT_1}_${INPUT_2}_${INPUT_3}/PORT_1a/"



mkdir -p ${DIRECTORY}"beersheba/" 
mkdir -p ${DIRECTORY}"isaura/" 
mkdir -p ${DIRECTORY}"output/" 
echo "Directories generated at: ${DIRECTORY}"



############################
# ISAURA MACRO AND RUNTIME
############################



## Generate IC config
CFG_MACRO_ISAURA="/gluster/data/next/files/TOPOLOGY_John/Configs/isa_new.LPR.NEXT100.${INPUT_1}_${INPUT_2}_${INPUT_3}_${NUMBER}.config"
# if it already exists, remove it, it'll cause problems if you don't!
rm -f $CFG_MACRO_ISAURA
echo "Macro started"


#### CONFIG FILE
# set paths
echo "files_in = '${INFILE}'  "                                 >> ${CFG_MACRO_ISAURA}
echo "file_out = '${ISAUFILE}'  "                                >> ${CFG_MACRO_ISAURA}

# compression library
echo "compression = 'ZLIB4' "                                   >> ${CFG_MACRO_ISAURA}
# max number of events to run
echo "event_range = all"                                        >> ${CFG_MACRO_ISAURA}
# run number 0 is for MC
echo "run_number = ${RUNNUMBER} "                               >> ${CFG_MACRO_ISAURA}
# detector database
echo "detector_db = 'next100' "                                 >> ${CFG_MACRO_ISAURA}
# How frequently to print events
echo "print_mod = 10 "                                           >> ${CFG_MACRO_ISAURA}

echo "paolina_params            = dict(
        vox_size                = ${VOXEL_SIZE},
        strict_vox_size         = False,
        energy_threshold        = 10 * keV,
        min_voxels              = 3,
        blob_radius             = ${BLOB_RAD},
        max_num_hits            = 10000)"                       >> ${CFG_MACRO_ISAURA}

export HDF5_USE_FILE_LOCKING='FALSE'
city isaura ${CFG_MACRO_ISAURA}

# if the last process (300), produce FOM
echo "Isaura done!"
if [ "$NUMBER" -eq 300 ]; then
	# pause out of fear that the other jobs aren't finished
	sleep 600
	echo "Producing FOM..."
	python3 /gluster/home/jwaiton/scripts/bin/cluster_FOM_calc.py $DIRECTORY
fi
