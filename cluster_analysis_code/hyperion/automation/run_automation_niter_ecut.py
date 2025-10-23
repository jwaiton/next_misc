# automation of niter and ecut across beersheba

import os
import subprocess
from multiprocessing import Pool
from functools import partial
import time as t
import shutil
import traceback


init_conda = f"""
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
"""

def beersheba_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          MC,
                          params):
    
    n_iter  = params['n_iter']
    ecut  = params['ecut']
    
    if MC:
        s_path   = 'N100_LPR_MC'
        psf_path = '/scratch/halmazan/NEXT/N100_LPR/lighttables/040925_full_25us.psf'
    else:
        s_path = 'N100_LPR'
        psf_path = '/scratch/halmazan/NEXT/N100_LPR/lighttables/040925_full_25us.psf'

    CONFIG_DIR=f"/scratch/halmazan/NEXT/{s_path}/{RUN}/configs/{CITY}-{TIMESTAMP}"


    slurm_content = f"""#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}

#SBATCH --error=errors/%x-%j-nI{n_iter}eC{ecut}-LDC{LDC}.err
#SBATCH --output=logs/%x-%j-nI{n_iter}eC{ecut}-LDC{LDC}.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL

{init_conda}

CITY=beersheba
RUN={RUN}
TIMESTAMP={TIMESTAMP}
SOPH_TIMESTAMP={TIMESTAMP}
LDC={LDC}

INPUT_DIR="/scratch/halmazan/NEXT/{s_path}/{RUN}/sophronia/{INPT_TIMESTAMP}/ldc{LDC}"
OUTPUT_DIR="/scratch/halmazan/NEXT/{s_path}/{RUN}/beersheba/{TIMESTAMP}/ldc{LDC}"
CONFIG_DIR="/scratch/halmazan/NEXT/{s_path}/{RUN}/configs/{CITY}-{TIMESTAMP}"

mkdir -p "$CONFIG_DIR"
mkdir -p "$OUTPUT_DIR"

CONFIG_LIST="/scratch/halmazan/NEXT/PROCESSING/thekla_configs/config_list_{CITY}_{RUN}_{TIMESTAMP}-LDC{LDC}.txt"
> "$CONFIG_LIST" # clear previous config lists


for file in "$INPUT_DIR"/*; do
	filename=$(basename "$file")
	raw_name="${{filename%.*}}"
	
	config_path="${{CONFIG_DIR}}/${{raw_name}}.conf"
	# create config here
	echo "files_in = '${{file}}'"                              > ${{config_path}}
	echo "file_out = '${{OUTPUT_DIR}}/${{raw_name}}_{CITY}.h5'" >> ${{config_path}}
	echo "compression = 'ZLIB4'"                             >> ${{config_path}}
	echo "event_range=1000"                                  >> ${{config_path}}
	echo "run_number = {RUN}"                               >> ${{config_path}}
	echo "detector_db = 'next100'"                           >> ${{config_path}}
	echo "print_mod = 1"                                     >> ${{config_path}}
	echo "threshold = 4"                                    >> ${{config_path}}
        echo "same_peak = True"                                  >> ${{config_path}}
    echo "deconv_params = dict(
        q_cut = 5,
        drop_dist = [16., 16., 4.],
        cluster_size = 5,
        psf_fname = '{psf_path}',
        e_cut = 0.2,
        n_iterations = 20,
        iteration_tol = 1e-09,
        sample_width = [15.55, 15.55],
        bin_size     = [1., 1.],
        energy_type  = Ec,
        diffusion = (1.0, 1.0),
        deconv_mode = joint,
        n_dim = 2,
        cut_type = rel,
        inter_method = cubic)"                              >> ${{config_path}}

	echo "satellite_params = None"                          >> ${{config_path}}
    echo "corrections = None"                               >> ${{config_path}}

	echo "${{config_path}}" >> "${{CONFIG_LIST}}"
done

echo "Config produced, initialising IC"
# Initialise IC
conda init bash
source /scratch/halmazan/NEXT/IC_alter-blob-centre/init_IC.sh
conda activate IC-3.8-2024-06-08

# parallel execution
cat "$CONFIG_LIST" | xargs -n 1 -P "$SLURM_CPUS_PER_TASK" -I{{}} city {CITY} {{}}

"""
    
    return slurm_content


def isaura_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          MC,
                          params):

    blobR  = params['blobR']
    scanR  = params['scanR']
    voxelS = params['voxS']    

    if MC:
        s_path   = 'N100_LPR_MC'

    else:
        s_path = 'N100_LPR'

    
    CONFIG_DIR=f"/scratch/halmazan/NEXT/N100_LPR/{RUN}/configs/{CITY}-{TIMESTAMP}"


    slurm_content = f"""#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}

#SBATCH --error=errors/%x-%j-bR{blobR}sR{scanR}vS{voxelS}-LDC{LDC}.err
#SBATCH --output=logs/%x-%j-bR{blobR}sR{scanR}vS{voxelS}-LDC{LDC}.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL

{init_conda}

CITY=isaura
RUN={RUN}
TIMESTAMP={TIMESTAMP}
SOPH_TIMESTAMP={TIMESTAMP}
LDC={LDC}
BLOB_RAD="{blobR} * mm"
VOXEL_SIZE="[{voxelS} * mm, {voxelS} * mm, {voxelS} * mm]"

INPUT_DIR="/data/halmazan/NEXT/{s_path}/{RUN}/beersheba/{TIMESTAMP}/ldc{LDC}"
OUTPUT_DIR="/scratch/halmazan/NEXT/{s_path}/{RUN}/isaura/{TIMESTAMP}/ldc{LDC}"
CONFIG_DIR="/scratch/halmazan/NEXT/{s_path}/{RUN}/configs/{CITY}-{TIMESTAMP}"

mkdir -p "$CONFIG_DIR"
mkdir -p "$OUTPUT_DIR"

CONFIG_LIST="/scratch/halmazan/NEXT/PROCESSING/thekla_configs/config_list_{CITY}_{RUN}_{TIMESTAMP}-LDC{LDC}.txt"
> "$CONFIG_LIST" # clear previous config lists


for file in "$INPUT_DIR"/*; do
	filename=$(basename "$file")
	raw_name="${{filename%.*}}"
	
	config_path="${{CONFIG_DIR}}/${{raw_name}}.conf"
	# create config here
	echo "files_in = '${{file}}'"                              > ${{config_path}}
	echo "file_out = '${{OUTPUT_DIR}}/${{raw_name}}_{CITY}.h5'" >> ${{config_path}}
	echo "compression = 'ZLIB4'"                             >> ${{config_path}}
	echo "event_range=1000"                                  >> ${{config_path}}
	echo "run_number = {RUN}"                               >> ${{config_path}}
	echo "detector_db = 'next100'"                           >> ${{config_path}}
        echo "print_mod = 100"                                 >> ${{config_path}}
	echo "paolina_params  = dict(
		vox_size = ${{VOXEL_SIZE}},
		strict_vox_size = False,
		energy_threshold = 10 * keV,
		min_voxels = 3,
		blob_radius = ${{BLOB_RAD}},
		max_num_hits = 1000000)"                    >> ${{config_path}}
	

	echo "${{config_path}}" >> "${{CONFIG_LIST}}"
done

echo "Config produced, initialising IC"
# Initialise IC
conda init bash
source /scratch/halmazan/NEXT/IC_alter-blob-centre/init_IC.sh
conda activate IC-3.8-2024-06-08

# parallel execution
cat "$CONFIG_LIST" | xargs -n 1 -P "$SLURM_CPUS_PER_TASK" -I{{}} city {CITY} {{}}

"""
    
    return slurm_content


def topology_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          params
                          ):
    
    # unpack the params
    blobR  = params['blobR']
    scanR  = params['scanR']
    voxelS = params['voxS']
    n_iter = params['n_iter']
    ecut = params['ecut']
    slurm_content = f"""#!/bin/bash
#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --qos=regular
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}

#SBATCH --error=errors_topology/%x-%j-bR{blobR}sR{scanR}vS{voxelS}-nI{n_iter}-eC{ecut}-LDC{LDC}.err
#SBATCH --output=logs_topology/%x-%j-bR{blobR}sR{scanR}vS{voxelS}-nI{n_iter}-eC{ecut}-LDC{LDC}.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL

{init_conda}


echo "Slurm job id is ${{SLURM_JOB_ID}}"


RUN={RUN}
TIMESTAMP={TIMESTAMP}
CITY={CITY}
z_lower=20
z_upper=1195
r_lim=450
e_lower=1.45
e_upper=1.75

echo "======================="
echo "RUN {RUN}"
echo "TIMESTAMP {TIMESTAMP}"
echo "CITY {CITY}"
echo "z limit [${{z_lower}}, ${{z_upper}}]"
echo "e limit [${{e_lower}}, ${{e_upper}}]"
echo "r limit ${{r_lim}}"
echo "======================="

conda init bash
source /scratch/halmazan/NEXT/IC_alter-blob-centre/init_IC.sh
conda activate IC-3.8-2024-06-08

python3 /scratch/halmazan/NEXT/PROCESSING/topology_cuts/bin/topology_checker_2209.py {RUN} {TIMESTAMP} ${{z_lower}} ${{z_upper}} ${{r_lim}} ${{e_lower}} ${{e_upper}} {CITY}

"""

    return slurm_content


def thekla_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          params):
    
    # unpack the params
    blobR  = params['blobR']
    scanR  = params['scanR']
    voxelS = params['voxelS']
    
    CONFIG_DIR=f"/scratch/halmazan/NEXT/N100_LPR/{RUN}/configs/{CITY}-{TIMESTAMP}"


    slurm_content = f"""#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}

#SBATCH --error=errors/%x-%j-bR{blobR}sR{scanR}vS{voxelS}-LDC{LDC}.err
#SBATCH --output=logs/%x-%j-bR{blobR}sR{scanR}vS{voxelS}-LDC{LDC}.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL

{init_conda}

CITY=thekla
RUN={RUN}
TIMESTAMP={TIMESTAMP}
SOPH_TIMESTAMP={TIMESTAMP}
LDC={LDC}

VOXEL_SIZE="[{voxelS} * mm, {voxelS} * mm, {voxelS} * mm]"
BLOB_RAD="{blobR} * mm"
SCAN_RAD="{scanR} * mm"

INPUT_DIR="/scratch/halmazan/NEXT/N100_LPR/{RUN}/sophronia/{INPT_TIMESTAMP}/ldc{LDC}"
OUTPUT_DIR="/scratch/halmazan/NEXT/N100_LPR/{RUN}/thekla/{TIMESTAMP}/ldc{LDC}"
CONFIG_DIR="/scratch/halmazan/NEXT/N100_LPR/{RUN}/configs/{CITY}-{TIMESTAMP}"

mkdir -p "$CONFIG_DIR"
mkdir -p "$OUTPUT_DIR"

CONFIG_LIST="/scratch/halmazan/NEXT/PROCESSING/thekla_configs/config_list_{CITY}_{RUN}_{TIMESTAMP}-LDC{LDC}.txt"
> "$CONFIG_LIST" # clear previous config lists


for file in "$INPUT_DIR"/*; do
	filename=$(basename "$file")
	raw_name="${{filename%.*}}"
	
	config_path="${{CONFIG_DIR}}/${{raw_name}}.conf"
	# create config here
	echo "files_in = '${{file}}'"                              > ${{config_path}}
	echo "file_out = '${{OUTPUT_DIR}}/${{raw_name}}_{CITY}.h5'" >> ${{config_path}}
	echo "compression = 'ZLIB4'"                             >> ${{config_path}}
	echo "event_range=1000"                                  >> ${{config_path}}
	echo "run_number = {RUN}"                               >> ${{config_path}}
	echo "detector_db = 'next100'"                           >> ${{config_path}}
	echo "print_mod = 1"                                     >> ${{config_path}}
	echo "threshold = 10"                                    >> ${{config_path}}
	echo "drop_distance = [16., 16., 4.]"                    >> ${{config_path}}
	echo "drop_minimum = 5"                                  >> ${{config_path}}
	echo "paolina_params  = dict(
		vox_size = ${{VOXEL_SIZE}},
		strict_vox_size = False,
		energy_threshold = 10 * keV,
		min_voxels = 3,
		blob_radius = ${{BLOB_RAD}},
		scan_radius = ${{SCAN_RAD}},
		max_num_hits = 1000000)"                    >> ${{config_path}}
	
	echo "corrections = None"      >> ${{config_path}}

	echo "${{config_path}}" >> "${{CONFIG_LIST}}"
done

echo "Config produced, initialising IC"
# Initialise IC
conda init bash
source /scratch/halmazan/NEXT/IC_alter-blob-centre/init_IC.sh
conda activate IC-3.8-2024-06-08

# parallel execution
cat "$CONFIG_LIST" | xargs -n 1 -P "$SLURM_CPUS_PER_TASK" -I{{}} city {CITY} {{}}

"""
    
    return slurm_content


def sophronia_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          MC = False):

    if MC:
        INPUT_CITY = 'hypathia'
        LPR_PATH   = 'N100_LPR_MC'
        RUN_CHECK  = -1
    else:
        INPUT_CITY = 'irene'
        LPR_PATH   = 'N100_LPR'
        RUN_CHECK  = RUN
    CONFIG_DIR=f"/scratch/halmazan/NEXT/LPR_PATH/{RUN}/configs/{CITY}-{TIMESTAMP}"

    slurm_content = f"""#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}

#SBATCH --error=errors/%x-%j-{CITY}-{INPUT_CITY}-LDC{LDC}.err
#SBATCH --output=logs/%x-%j-{CITY}-{INPUT_CITY}-LDC{LDC}.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL

{init_conda}

CITY={CITY}
RUN={RUN}
TIMESTAMP={TIMESTAMP}
SOPH_TIMESTAMP={TIMESTAMP}
LDC={LDC}

INPUT_DIR="/data/halmazan/NEXT/{LPR_PATH}/{RUN}/{INPUT_CITY}/{TIMESTAMP}/ldc{LDC}"
OUTPUT_DIR="/scratch/halmazan/NEXT/{LPR_PATH}/{RUN}/{CITY}/{TIMESTAMP}/ldc{LDC}"
CONFIG_DIR="/scratch/halmazan/NEXT/{LPR_PATH}/{RUN}/configs/{CITY}-{TIMESTAMP}"

mkdir -pv "$CONFIG_DIR"
mkdir -pv "$OUTPUT_DIR"

CONFIG_LIST="/scratch/halmazan/NEXT/PROCESSING/thekla_configs/config_list_{CITY}_{RUN}_{TIMESTAMP}-LDC{LDC}.txt"
> "$CONFIG_LIST" # clear previous config lists

for file in "$INPUT_DIR"/*; do
	filename=$(basename "$file")
	raw_name="${{filename%.*}}"

	config_path="${{CONFIG_DIR}}/${{raw_name}}.conf"    
	echo "files_in = '${{file}}'"                               > ${{config_path}}
	echo "file_out = '${{OUTPUT_DIR}}/${{raw_name}}_{CITY}.h5'" >> ${{config_path}}
    echo "compression = 'ZLIB4'"                                >> ${{config_path}}
    echo "run_number  = {RUN_CHECK}"                                  >> ${{config_path}}
    echo "detector_db = 'next100'"                              >> ${{config_path}}
    echo "print_mod   = 100"                                    >> ${{config_path}}
    echo "event_range = all"                                    >> ${{config_path}}

    echo "drift_v     = 0.87 * mm / mus"                        >> ${{config_path}}

    echo "s1_params = dict(
    s1_nmin     =    1,
    s1_nmax     =    1,
    s1_emin     =    0 * pes,
    s1_emax     =  1e6 * pes,
    s1_wmin     =   50 * ns,
    s1_wmax     =  500 * ns,
    s1_hmin     =    0 * pes,
    s1_hmax     =  1e6 * pes,
    s1_ethr     =    0 * pes,
    )"                                                          >> ${{config_path}}

    echo "s2_params = dict(
    s2_nmin     =    1,
    s2_nmax     =    1,
    s2_emin     =    0 * pes,
    s2_emax     =  1e8 * pes,
    s2_wmin     =    1 * mus,
    s2_wmax     = 1000 * mus,
    s2_hmin     =    0 * pes,
    s2_hmax     =  1e5 * pes,
    s2_ethr     =    0 * pes,
    s2_nsipmmin =    1,
    s2_nsipmmax = 9000,
    )"                                                          >> ${{config_path}}

    echo "rebin = 10000000"                                     >> ${{config_path}}
    echo "rebin_method = stride"                                >> ${{config_path}}

    echo "sipm_charge_type = raw"                               >> ${{config_path}}
    echo "#fiducial_fraction = 0.95"                            >> ${{config_path}}

    echo "global_reco_algo = barycenter"                        >> ${{config_path}}
    echo "global_reco_params = dict(Qthr = 10 * pes)"           >> ${{config_path}}

    echo "same_peak = True"                                     >> ${{config_path}}


    echo "q_thr = 5 * pes"                                      >> ${{config_path}}

    echo "${{config_path}}" >> "${{CONFIG_LIST}}"
done

echo "Config produced, initialising IC"
# Initialise IC
conda init bash
source /scratch/halmazan/NEXT/IC_alter-blob-centre/init_IC.sh
conda activate IC-3.8-2024-06-08

# parallel execution
cat "$CONFIG_LIST" | xargs -n 1 -P "$SLURM_CPUS_PER_TASK" -I{{}} city {CITY} {{}}


"""
    return slurm_content


def create_slurm_file(JOB_TYPE,
                      RUN,
                      TIMESTAMP,
                      INPT_TIMESTAMP,
                      LDC,
                      CITY,
                      job_name, 
                      script_path, 
                      output_path, 
                      error_path,
                      partition, 
                      time, 
                      nodes, 
                      ntasks, 
                      cpus_per_task, 
                      mem,
                      MC,
                      params):
    
    if JOB_TYPE == 'thekla':
        slurm_content = thekla_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          params)
    elif JOB_TYPE == 'topology':
        slurm_content = topology_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          params)
    elif JOB_TYPE == 'beersheba':
        slurm_content = beersheba_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          MC,
                          params)
    elif JOB_TYPE == 'isaura':
        slurm_content = isaura_slurm_template(RUN,
                  TIMESTAMP,
                  INPT_TIMESTAMP,
                  LDC,
                  CITY,
                  job_name, 
                  script_path, 
                  output_path, 
                  error_path,
                  partition, 
                  time, 
                  nodes, 
                  ntasks, 
                  cpus_per_task, 
                  mem,
                  MC,
                  params)
    elif JOB_TYPE == 'sophronia':
        slurm_content = sophronia_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition, 
                          time, 
                          nodes, 
                          ntasks, 
                          cpus_per_task, 
                          mem,
                          MC)



    slurm_file = f"slurm_files/{job_name}.slurm"
    with open(slurm_file, "w") as file:
        file.write(slurm_content)
    return slurm_file



def create_and_submit(LDC,
                      JOB_TYPE,
                      RUN,
                      TIMESTAMP,
                      INPT_TIMESTAMP,
                      CITY,
                      job_name, 
                      script_path, 
                      output_path, 
                      error_path,
                      partition, 
                      time, 
                      nodes, 
                      ntasks, 
                      cpus_per_task, 
                      mem,
                      MC,
                      params):
    
    job_name_ext = f'{job_name}-{LDC}'

    slurm_file = create_slurm_file(JOB_TYPE,
                      RUN,
                      TIMESTAMP,
                      INPT_TIMESTAMP,
                      LDC,
                      CITY,
                      job_name_ext, 
                      script_path, 
                      output_path, 
                      error_path,
                      partition, 
                      time, 
                      nodes, 
                      ntasks, 
                      cpus_per_task, 
                      mem,
                      MC,
                      params)
    
    submit_slurm_job(slurm_file)


def submit_slurm_job(slurm_file):
    try:
        result = subprocess.run(["sbatch", slurm_file], check=True, capture_output=True, text=True)
        print(f"Job submitted successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")


def get_running_jobs():
    try:
        result = subprocess.run(["squeue", "-u", os.getenv("USER")], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        running_jobs = len(lines) - 1  # Subtract header line
        print(f"Currently running jobs: {running_jobs}")
        return running_jobs
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving running jobs: {e.stderr}")
        return 0


def run_city_soph(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC):

    script_path     = ''
    error_path      = f'errors_soph/%x-%j-%A-%a-{TIMESTAMP}.err'
    output_path     = f'logs_soph/%x-%j-%A-%a-{TIMESTAMP}.log'
    partition       ="general" 
    time            ="24:00:00"
    nodes           =1
    ntasks          =1
    cpus_per_task   =36
    mem             ="32G"   


    print(f'==========================')
    print(f'==========================')
    print(f'    SOPHRONIA RUNTIME')
    print(f'==========================')
    print(f'==========================')

    # running thekla
    for i, RN in enumerate(RUNS):
        # give it a moment to catch up
        t.sleep(30)
        while get_running_jobs() > 3:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)

        job_name        = f'{CITY}-{RN}-{TIMESTAMP}-'

        process_ldc = partial(
            create_and_submit,
            RUN = RN,
            job_name=job_name,
            JOB_TYPE=JOB_TYPE,
            TIMESTAMP=TIMESTAMP,
            INPT_TIMESTAMP=CORR_TS,
            CITY="sophronia",   # BE CAREFUL
            script_path=script_path,
            output_path=output_path,
            error_path=error_path,
            partition=partition,
            time=time,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            mem=mem,
            MC=MC,
            params = {}
        )

        print(f'==========================')
        print(f'Submitting pool for {RN}')
        print(f'==========================')
        with Pool() as pool:
            pool.map(process_ldc, range(1, 8))

        # Take a break in between processes, there should be NO jobs running here
        t.sleep(30)
        while get_running_jobs() != 0:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)


def run_city_beersheba(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params):


    script_path     = ''
    error_path      = f'errors_beer/%x-%j-%A-%a-{TIMESTAMP}.err'
    output_path     = f'logs_beer/%x-%j-%A-%a-{TIMESTAMP}.log'
    partition       ="general" 
    time            ="24:00:00"
    nodes           =1
    ntasks          =1
    cpus_per_task   =36
    mem             ="26G"   

    n_iter = params['n_iter']
    ecut   = params['ecut']

    print(f'==========================')
    print(f'==========================')
    print(f'    BEERSHEBA RUNTIME')
    print(f'==========================')
    print(f'==========================')

    # running thekla
    for i, RN in enumerate(RUNS):
        # give it a moment to catch up
        t.sleep(30)
        while get_running_jobs() > 3:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)
        
        job_name        = f'{CITY}-{RN}-{TIMESTAMP}-nI{n_iter}-eC{ecut}'

        process_ldc = partial(
            create_and_submit,
            RUN = RN,
            job_name=job_name,
            JOB_TYPE=JOB_TYPE,
            TIMESTAMP=TIMESTAMP,
            INPT_TIMESTAMP=CORR_TS,
            CITY="beersheba",
            script_path=script_path,
            output_path=output_path,
            error_path=error_path,
            partition=partition,
            time=time,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            mem=mem,
            MC=MC,
            params = params)

        print(f'==========================')
        print(f'Submitting pool for {RN}')
        print(f'==========================')
        with Pool() as pool:
            pool.map(process_ldc, range(1, 8))
    
    
        # Take a break in between processes, there should be NO jobs running here
        t.sleep(30)
        while get_running_jobs() != 0:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)

        print('Moving beersheba files to data...')
        resolve_data(RN, TIMESTAMP, 'beersheba')


def run_city_thekla(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, blobR, scanR, voxelS):

    script_path     = ''
    error_path      = f'errors_thekla/%x-%j-%A-%a-{TIMESTAMP}.err'
    output_path     = f'logs_thekla/%x-%j-%A-%a-{TIMESTAMP}.log'
    partition       ="general" 
    time            ="24:00:00"
    nodes           =1
    ntasks          =1
    cpus_per_task   =36
    mem             ="32G"
    #blobR           =35
    #scanR           =45
    #voxelS          =15    

    print(f'==========================')
    print(f'==========================')
    print(f'     THEKLA RUNTIME')
    print(f'==========================')
    print(f'==========================')


    # running thekla
    for i, RN in enumerate(RUNS):

        # give it a moment to catch up
        t.sleep(30)
        while get_running_jobs() > 3:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)

        job_name        = f'{CITY}-{RN}-{TIMESTAMP}-bR{blobR}-sR{scanR}-vS{voxelS}'

        process_ldc = partial(
            create_and_submit,
            RUN = RN,
            job_name=job_name,
            JOB_TYPE=JOB_TYPE,
            TIMESTAMP=TIMESTAMP,
            INPT_TIMESTAMP=CORR_TS,
            CITY="thekla",
            script_path=script_path,
            output_path=output_path,
            error_path=error_path,
            partition=partition,
            time=time,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            mem=mem,
            blobR=blobR,
            scanR=scanR,
            voxelS=voxelS
        )

        print(f'==========================')
        print(f'Submitting pool for {RN}')
        print(f'==========================')
        with Pool() as pool:
            pool.map(process_ldc, range(1, 8))
        
        
        # Take a break in between processes, there should be NO jobs running here
        t.sleep(30)
        while get_running_jobs() != 0:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)


def run_city_isaura(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params):
    JOB_TYPE        = 'isaura'
    CITY            = 'isaura'
    
    script_path     = ''
    error_path      = f'errors_isaura/%x-%j-%A-%a-{TIMESTAMP}.err'
    output_path     = f'logs_isaura/%x-%j-%A-%a-{TIMESTAMP}.log'
    partition       ="general" 
    time            ="24:00:00"
    nodes           =1
    ntasks          =1
    cpus_per_task   =36
    mem             ="32G"
    #blobR           =35
    #scanR           =45
    #voxelS          =15    
    
    # unpack
    blobR = params['blobR']    
    scanR = params['scanR']
    voxS = params['voxS']

    # running thekla
    for i, RN in enumerate(RUNS):

        # give it a moment to catch up
        t.sleep(30)
        while get_running_jobs() > 3:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)

        job_name        = f'{CITY}-{RN}-{TIMESTAMP}-bR{blobR}-sR{scanR}-vS{voxS}'

        process_ldc = partial(
            create_and_submit,
            RUN = RN,
            job_name=job_name,
            JOB_TYPE=JOB_TYPE,
            TIMESTAMP=TIMESTAMP,
            INPT_TIMESTAMP=CORR_TS,
            CITY="isaura",
            script_path=script_path,
            output_path=output_path,
            error_path=error_path,
            partition=partition,
            time=time,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            mem=mem,
            MC=MC,
            params = params
        )

        print(f'==========================')
        print(f'Submitting pool for {RN}')
        print(f'==========================')
        with Pool() as pool:
            pool.map(process_ldc, range(1, 8))


        # Take a break in between processes, there should be NO jobs running here
        t.sleep(30)
        while get_running_jobs() != 0:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)

        print('Moving isaura files to data...')
        resolve_data(RN, TIMESTAMP, 'isaura')

def run_city_topology(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params):

    print(f'==========================')
    print(f'==========================')
    print(f'     TOPOLOGY RUNTIME')
    print(f'==========================')
    print(f'==========================')

    blobR = params['blobR']    
    scanR = params['scanR']
    voxS = params['voxS']


    JOB_TYPE        = 'topology'
    
    script_path     = ''
    error_path      = f'errors/%x-%j-%A-%a-{TIMESTAMP}.err'
    output_path     = f'logs/%x-%j-%A-%a-{TIMESTAMP}.log'
    partition       ="general" 
    time            ="24:00:00"
    nodes           =1
    ntasks          =1
    cpus_per_task   =6
    mem             ="20G" 
    LDC = 1 # since we dont care


    # running topology cut
    for i, RN in enumerate(RUNS):
        job_name        = f'CORRECTIONS-{RN}-{TIMESTAMP}-bR{blobR}-sR{scanR}-vS{voxS}'

        # compiles all the LDCs, so just set LDC to one for this one
        create_and_submit(LDC = LDC,
            RUN = RN,
            job_name=job_name,
            JOB_TYPE=JOB_TYPE,
            TIMESTAMP=TIMESTAMP,
            INPT_TIMESTAMP=CORR_TS,
            CITY="isaura",
            script_path=script_path,
            output_path=output_path,
            error_path=error_path,
            partition=partition,
            time=time,
            nodes=nodes,
            ntasks=ntasks,
            cpus_per_task=cpus_per_task,
            mem=mem,
            MC=MC,
            params=params)

        # give it a moment to catch up
        t.sleep(30)
        while get_running_jobs() > 3:
            print('Prior jobs not finished, waiting 5 minutes...')
            t.sleep(300)



def run_city(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, blobR, scanR, voxelS):

    
    if CITY == 'sophronia':
        run_city_soph(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC)


def run_full_param_scan(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params):
    
    '''
    For the time being this function should:
        run beersheba for a given set of parameters etc
    '''

    # Take a break in between processes, there should be NO jobs running here
    t.sleep(30)
    while get_running_jobs() != 0:
        print('Prior jobs not finished, waiting 5 minutes...')
        t.sleep(300)
    
    JOB_TYPE = 'beersheba'
    city = 'beersheba'
    run_city_beersheba(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params)


    JOB_TYPE = 'isaura'
    city = 'isaura'
    run_city_isaura(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params)
    ##run_city_soph(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC)


    # set this to isaura to look within /data/
    JOB_TYPE = 'isaura'
    
    run_city_topology(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params)
    #run_city_soph(TIMESTAMP, RUNS, CITY, JOB_TYPE, CORR_TS, MC)

    # Take a break in between processes, there should be NO jobs running here
    t.sleep(30)
    while get_running_jobs() != 0:
        print('Prior jobs not finished, waiting 5 minutes...')
        t.sleep(300)
   

def resolve_data(RUNS, TIMESTAMPS, CITY):
    '''
    moves all the relevant data from scratch to data to save space.

    '''
    
    rn = RUNS
    ts = TIMESTAMPS
    # data from correct and cut:
    #for rn, ts in zip(RUNS, TIMESTAMPS):
    folder_out = f'/scratch/halmazan/NEXT/N100_LPR/{rn}/{CITY}/{ts}/'
    placeholder_folder = f'/data/halmazan/NEXT/N100_LPR/{rn}/{CITY}/{ts}'
    for root, dirs, files in os.walk(folder_out):
        for file in files:
            try:
                src_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_path, folder_out)
                dest_path = os.path.join(placeholder_folder, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.move(src_path, dest_path)
            except Exception as e:
                print(f'Moving the file {file} broke!')
                print(traceback.format_exc())
                
                


def main():


    # assign the timestamps, and corresponding blob radii here, then run them
    # order of timestamp is 'blobRscanRvoxS'
    # But we'll be changing this for beersheba/isaura most likely.
    TSs             = [1025, 1035, 2005, 2015, 2025, 2035, 3005, 3015, 3025, 3035, 4005, 4015, 4025, 4035]
    #TSs             = [1005, 1015]
    #RUNS            = ['202510']
    JOB_TYPE        = 'beersheba'
    
    #RUNS            = [22222]
    #CORR_TS         = 22222
    #RUNS           = [15589, 15590, 15591, 15592, 15593, 15594, 15596, 15597
    
    RUNS            = [15589, 15590, 15591, 15592]
    
    # MC STANDERD 11/07/25
    #CORR_TS         = 202510 # The corrected sophronia data timestamp
    
    CORR_TS          = 230725

    CITY            = 'beersheba'
    MC              = False

    # the test
    #TSs = [375015, 476015]

    for TS in TSs:
        # split into three if you've setup timestamps that way
        #blobR, scanR, voxS = int(str(TS)[:2]), int(str(TS)[2:4]), int(str(TS)[4:])
        n_iter, ecut = int(str(TS)[:2]), float(str(TS)[2:])
        ecut = ecut/100
        # hard set blobR, scanR, and voxelS here
        voxS  = 18
        scanR = 50
        blobR = 35
        # define your relevant parameters here
        params = {'blobR'  : blobR,
                      'scanR'  : scanR,
                      'voxS'   : voxS,
                      'n_iter' : n_iter,
                      'ecut'   : ecut}

        print(f'\n\n\n\n\n\n\n\n\n')
        print("Running with parameters:")
        for key, value in params.items():
            print(f"  {key.capitalize()}: {value}")
        print(f"  Timestamp: {TS}")
        print(f"  Runs: {RUNS}")
        print(f"  Job Type: {JOB_TYPE}")
        print(f"  City: {CITY}")
        print(f"  Corrected Timestamp: {CORR_TS}")
        print(f"  Monte Carlo: {MC}")


        run_full_param_scan(TS, RUNS, CITY, JOB_TYPE, CORR_TS, MC, params)


main()
