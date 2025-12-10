echo "Initialise conda"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/halmazan/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/halmazan/miniconda/etc/profile.d/conda.sh" ]; then
        . "/home/halmazan/miniconda/etc/profile.d/conda.sh"
    else
        export PATH="/home/halmazan/miniconda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Read arguments into variables
RN=$1
TS=$2
blobR=$3
scanR=$4
voxel_size=$5
LDC=$6
CITY='thekla'

VOXEL_SIZE="[${voxel_size} * mm, ${voxel_size} * mm, ${voxel_size} * mm]"
BLOB_RAD="${blobR} * mm"
SCAN_RAD="${scanR} * mm"

INPUT_DIR="/scratch/halmazan/NEXT/N100_LPR/${RN}/sophronia/${TS}/ldc${LDC}"
OUTPUT_DIR="/scratch/halmazan/NEXT/N100_LPR/${RN}/thekla/${TS}/ldc${LDC}"
CONFIG_DIR="/scratch/halmazan/NEXT/N100_LPR/${RN}/configs/${CITY}-${TS}"

mkdir -p "$CONFIG_DIR"
mkdir -p "$OUTPUT_DIR"


CONFIG_LIST="/scratch/halmazan/NEXT/PROCESSING/thekla_configs/config_list_${CITY}_${RN}_${TS}-LDC${LDC}.txt"
> "$CONFIG_LIST" # clear previous config lists

for file in "$INPUT_DIR"/*; do
	filename=$(basename "$file")
	raw_name="${filename%.*}"
	
	config_path="${CONFIG_DIR}/${raw_name}.conf"
	# create config here
	echo "files_in = '${file}'"                              > ${config_path}
	echo "file_out = '${OUTPUT_DIR}/${raw_name}_${CITY}.h5'" >> ${config_path}
	echo "compression = 'ZLIB4'"                             >> ${config_path}
	echo "event_range=1000"                                  >> ${config_path}
	echo "run_number = ${RN}"                                >> ${config_path}
	echo "detector_db = 'next100'"                           >> ${config_path}
	echo "print_mod = 1"                                     >> ${config_path}
	echo "threshold = 10"                                    >> ${config_path}
	echo "drop_distance = [16., 16., 4.]"                    >> ${config_path}
	echo "drop_minimum = 5"                                  >> ${config_path}
	echo "paolina_params  = dict(
		vox_size = ${VOXEL_SIZE},
		strict_vox_size = False,
		energy_threshold = 10 * keV,
		min_voxels = 3,
		blob_radius = ${BLOB_RAD},
		scan_radius = ${SCAN_RAD},
		max_num_hits = 1000000)"                    >> ${config_path}
	
	echo "corrections = None"      >> ${config_path}

	echo "${config_path}" >> "${CONFIG_LIST}"
done

echo "Config produced, initialising IC"
# Initialise IC
conda init bash
source /scratch/halmazan/NEXT/IC_alter-blob-centre/init_IC.sh
conda activate IC-3.8-2024-06-08

echo "===================================================="
echo "Number of processors available for crunching thekla:"
echo "$(nproc)"
echo "If this isn't enough, assign more in the slurm task!"
echo "===================================================="

# parallel execution
cat "$CONFIG_LIST" | xargs -n 1 -P "$(nproc)" -I{} city ${CITY} {}
