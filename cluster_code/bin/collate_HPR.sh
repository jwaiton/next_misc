NUMBER=$(($1+1))
RUNNUMBER=0

INPUT_1=$2
INPUT_2=$3
INPUT_3=$4

PORT="${INPUT_3}"

echo "Applying e_cut: ${INPUT_2}"
echo "Applying n_iter: ${INPUT_1}"
echo "Across port: ${INPUT_3}"

######################
# SOPHRONIA PARAMETERS
######################

REBIN=3
Q_THR=4

######################
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

#####################
# ISAURA PARAMETERS
#####################

VOXEL_SIZE="[12 * mm, 12 * mm, 12 * mm]"
BLOB_RAD="18 * mm"

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
# SOPHRONIA MACRO AND RUNTIME
############################

## Generate IC config
CFG_MACRO_SOPH="/gluster/data/next/files/TOPOLOGY_John/Configs/sophronia.HPR.NEXT100.${E_CUT}_${N_ITER}_${NUMBER}.config"

echo "Sophronia macro started"

## Correction file
PSF_MAP="/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/LightTables/map_NEXT100_MC.h5"
echo "Lighttable imported"

#### CONFIG FILE
# set paths
echo "files_in = '${INFILE}'  "                          	> ${CFG_MACRO_SOPH}
echo "file_out = '${SOPHFILE}'  "                          	>> ${CFG_MACRO_SOPH}
# compression library
echo "compression = 'ZLIB4' "                                   >> ${CFG_MACRO_SOPH}
# max number of events to run
echo "event_range = all"                                 	>> ${CFG_MACRO_SOPH}
# run number 0 is for MC
echo "run_number = ${RUNNUMBER} "                               >> ${CFG_MACRO_SOPH}
# detector database
echo "detector_db = 'next100' "					>> ${CFG_MACRO_SOPH}
# How frequently to print events
echo "print_mod = 50 "                                           >> ${CFG_MACRO_SOPH}

# drift velocity of particles
echo "drift_v = 1.05 * mm / mus "                                           >> ${CFG_MACRO_SOPH}

echo "s1_params = dict(
    s1_nmin     =   1      , # Min number of S1 signals
    s1_nmax     =   5      , # Max number of S1 signals
    s1_emin     =   5 * pes, # Min S1 energy integral
    s1_emax     =   1e4 * pes, # Max S1 energy integral
    s1_wmin     =   75 * ns , # Min width for S1
    s1_wmax     =   2 * mus , # Max width
    s1_hmin     =   2 * pes, # Min S1 height
    s1_hmax     =   1e4 * pes, # Max S1 height
    s1_ethr     =   0 * pes, # Energy threshold for S1
)"				>> ${CFG_MACRO_SOPH}

echo "s2_params = dict(
    s2_nmin     =   1      , # Min number of S2 signals
    s2_nmax     =   5      , # Max number of S2 signals
    s2_emin     =   100 * pes, # Min S2 energy integral
    s2_emax     =   1e9 * pes, # Max S2 energy integral in pes
    s2_wmin     =   0.5 * mus, # Min width
    s2_wmax     =   1000 * mus , # Max width
    s2_hmin     =   100 * pes, # Min S2 height
    s2_hmax     =   1e9 * pes, # Max S2 height
    s2_nsipmmin =   1      , # Min number of SiPMs touched
    s2_nsipmmax =   3000      , # Max number of SiPMs touched
    s2_ethr     =   0 * pes, # Energy threshold for S2
)"				>> ${CFG_MACRO_SOPH}

echo "rebin = ${REBIN} "                                           >> ${CFG_MACRO_SOPH}
echo "rebin_method = stride "                                           >> ${CFG_MACRO_SOPH}

echo "sipm_charge_type = raw "                                           >> ${CFG_MACRO_SOPH}

echo "q_thr = ${Q_THR} * pes "                                           >> ${CFG_MACRO_SOPH}

echo "global_reco_algo = barycenter "                                           >> ${CFG_MACRO_SOPH}
# be careful here, previously set to 20, but im unsure of its influence on the results...
echo "global_reco_params = dict(Qthr = ${Q_THR} * pes) "                                           >> ${CFG_MACRO_SOPH}

echo "same_peak = True "                                           >> ${CFG_MACRO_SOPH}

echo "corrections_file = '/gluster/data/next/files/TOPOLOGY_John/HPR_PARAMETER_CHECK/LightTables/map_NEXT100_MC.h5' "  >> ${CFG_MACRO_SOPH}
echo "apply_temp = False "                                           >> ${CFG_MACRO_SOPH}

export HDF5_USE_FILE_LOCKING='FALSE'
city sophronia ${CFG_MACRO_SOPH}

echo "Sophronia done! Beersheba running now"
# out of fear, there is a 5s sleep here
sleep 5s
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

echo "Beersheba done! Isaura running now"
# out of fear, we're adding a 5 second sleep here
sleep 1s
############################
# ISAURA MACRO AND RUNTIME
############################



## Generate IC config
CFG_MACRO_ISAURA="/gluster/data/next/files/TOPOLOGY_John/Configs/isaura.HPR.NEXT100.${E_CUT}_${N_ITER}_${NUMBER}.config"
echo "Macro started"


#### CONFIG FILE
# set paths
echo "files_in = '${BEERFILE}'  "                                 > ${CFG_MACRO_ISAURA}
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
if [ "$NUMBER" -eq 660 ]; then
	# pause out of fear that the other jobs aren't finished
	sleep 600
	echo "Producing FOM..."
	python3 /gluster/home/jwaiton/scripts/bin/cluster_FOM_calc.py $DIRECTORY
fi
