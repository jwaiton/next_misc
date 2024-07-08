NUMBER=$(($1+1))
RUNNUMBER=0

INPUT_1=$2
INPUT_2=$3
INPUT_3=$4

PORT="${INPUT_3}"

echo "Applying e_cut: ${INPUT_2}"
echo "Applying n_iter: ${INPUT_1}"
echo "Across port: ${INPUT_3}"

#####################
# BEERSHEBA PARAMETERS
######################

E_CUT="${INPUT_2}"			# cut to voxel values of deconv output
CUT_TYPE="abs"			# absolute or relative cut-type

N_ITER=${INPUT_1}
ITER_TOL="1e-10"

SAMPLE_WIDTH="[15.55, 15.55]"	# SiPM pitch
BIN_SIZE="[1., 1.]"		# interpolated bin size in mm

Q_CUT=4
DROP_DIST="[16., 16.]"		# Distance provided to 'drop_sensors'

####################
# FILE LOCATIONS
####################


INFILE="/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/hypathia_data/PORT_${PORT}/hypathia_${NUMBER}_208Tl.h5"
SOPHFILE="/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/data/PORT_${PORT}/sophronia/sophronia_${NUMBER}_208Tl.h5"
BEERFILE="/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/data/PORT_${PORT}/beersheba/beersheba_${NUMBER}_208Tl.h5"
ISAUFILE="/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/data/PORT_${PORT}/isaura/isaura_${NUMBER}_208Tl.h5"

# Activating IC

echo "IO files found"
## Configure scisoft software products
source /gluster/data/next/software/IC_sophronia/config_ic.sh
echo "IC sourced"

# for running the FOM
DIRECTORY="/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/data/PORT_${PORT}/"
#############################
# BEERSHEBA MACRO AND RUNTIME
############################

## Generate IC config
CFG_MACRO="/gluster/data/next/files/TOPOLOGY_John/Configs/sophronia.HPR.NEXT100.${E_CUT}_${N_ITER}_${NUMBER}.config"
# if it already exists, remove it, it'll cause problems if you don't!
#rm -f $CFG_MACRO
echo "Beersheba macro started"

## Corrections Paths
PSF_MAP="/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/LightTables/NEXT100_PSF_kr83m.h5"
echo "Lighttable imported"

#### CONFIG FILE
# set paths
echo "files_in = '${SOPHFILE}'  "                          	> ${CFG_MACRO}
echo "file_out = '${BEERFILE}'  "                          	>> ${CFG_MACRO}

# compression library
echo "compression = 'ZLIB4' "                                   >> ${CFG_MACRO}
# max number of events to run
echo "event_range = all"                                 	>> ${CFG_MACRO}
# run number 0 is for MC
echo "run_number = ${RUNNUMBER} "                               >> ${CFG_MACRO}
# detector database
echo "detector_db = 'next100' "					>> ${CFG_MACRO}
# How frequently to print events
echo "print_mod = 50 "                                           >> ${CFG_MACRO}

echo "same_peak = True"						>> ${CFG_MACRO}

echo "threshold = 4 * pes"					>> ${CFG_MACRO}

echo "deconv_params 	= dict(
	q_cut 		= ${Q_CUT},
	drop_dist	= ${DROP_DIST},
	psf_fname	= '${PSF_MAP}',
	e_cut		= ${E_CUT},
	n_iterations	= ${N_ITER},
	iteration_tol	= ${ITER_TOL},
	sample_width	= ${SAMPLE_WIDTH},
	bin_size	= ${BIN_SIZE},
	energy_type	= Ec,
	diffusion	= (1.0, 1.0),
	deconv_mode	= joint,
	n_dim		= 2,
	cut_type	= ${CUT_TYPE},
	inter_method	= cubic)"				>> ${CFG_MACRO}


export HDF5_USE_FILE_LOCKING='FALSE'
city beersheba ${CFG_MACRO}

echo "Beersheba done!"

