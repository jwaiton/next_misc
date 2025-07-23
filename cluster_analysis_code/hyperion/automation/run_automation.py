import os
import subprocess
from multiprocessing import Pool
from functools import partial
import time as t

### A remake of the mother script (automate_processing.py)
# this will do all the same things, but through a login node on tmux
# so that we can schedule things properly.
# first test is checking if we can schedule jobs from here.



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



def thekla_slurm_template(RUN,
                          TIMESTAMP,
                          INPT_TIMESTAMP,
                          LDC,
                          CITY,
                          job_name, 
                          script_path, 
                          output_path, 
                          error_path,
                          partition="general", 
                          time="01:00:00", 
                          nodes=1, 
                          ntasks=1, 
                          cpus_per_task=36, 
                          mem="32G",
                          blobR=35,
                          scanR=45,
                          voxelS=60):
    
    CONFIG_DIR=f"/scratch/halmazan/NEXT/N100_LPR/{RUN}/configs/{CITY}-{TIMESTAMP}"


    slurm_content = f"""#!/bin/bash
#SBATCH --qos=regular
#SBATCH --job-name={job_name}
#SBATCH --output={output_path}
#SBATCH --error={error_path}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}

#SBATCH --error=errors/%x-%j.err
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL

{init_conda}

CITY=thekla
RUN={RUN}
TIMESTAMP={TIMESTAMP}
SOPH_TIMESTAMP={TIMESTAMP}
LDC={LDC}

VOXEL_SIZE="[{voxelS} * mm, {voxelS} * mm, {voxelS} * mm]"
BLOR_RAD="{blobR} * mm"
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
	echo "run_number = ${RUN}"                               >> ${{config_path}}
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
                      partition="general", 
                      time="01:00:00", 
                      nodes=1, 
                      ntasks=1, 
                      cpus_per_task=36, 
                      mem="32G",
                      blobR=35,
                      scanR=45,
                      voxelS=60):
    
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
                          blobR,
                          scanR,
                          voxelS)
        
    slurm_file = f"{job_name}.slurm"
    with open(slurm_file, "w") as file:
        file.write(slurm_content)
    return slurm_file


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


### Example usage
##job_name = "example_job"
##script_path = "/path/to/your_script.py"
##output_path = "/scratch/halmazan/NEXT/PROCESSING/testing/output.log"
##error_path = "/scratch/halmazan/NEXT/PROCESSING/testing/error.log"
##
##slurm_file = create_slurm_file(JOB_TYPE
##                      RUN,
##                      TIMESTAMP,
##                      INPT_TIMESTAMP
##                      LDC,
##                      CITY,
##                      job_name, 
##                      script_path, 
##                      output_path, 
##                      error_path,
##                      partition="general", 
##                      time="01:00:00", 
##                      nodes=1, 
##                      ntasks=1, 
##                      cpus_per_task=36, 
##                      mem="32G",
##                      blobR=35,
##                      scanR=45,
##                      voxelS=60)
##submit_slurm_job(slurm_file)


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
                      blobR,
                      scanR,
                      voxelS):
    
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
                      blobR,
                      scanR,
                      voxelS)
    
    submit_slurm_job(slurm_file)


def main():
    
    '''
    For the time being this function should:
        run thekla for a given blob radius, set of parameters etc
    '''
    JOB_TYPE        = 'thekla'
    RUNS            = [15589, 15590, 15591, 15592, 15593, 15594, 15596, 15597]
    TIMESTAMP       = 240725
    CORR_TS         = 230725
    CITY            = 'thekla'
    
    script_path     = ''
    error_path      = f'errors/%x-%j-%A-%a-{TIMESTAMP}.err'
    output_path     = f'logs/%x-%j-%A-%a-{TIMESTAMP}.log'
    partition       ="general" 
    time            ="24:00:00"
    nodes           =1
    ntasks          =1
    cpus_per_task   =36
    mem             ="32G"
    blobR           =35
    scanR           =45
    voxelS          =60    
    
    
    
    print('Trying to run thekla')



    for i, RN in enumerate(RUNS):

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


        print(f'Submitting pool for {RN}')
        with Pool() as pool:
            pool.map(process_ldc, range(1, 8))
        
        # give it a moment to catch up
        t.sleep(30)
        while get_running_jobs() > 3:
            print('Prior jobs not finished, not rushing')
            t.sleep(60)

        # early checker
        if i > 2:
            break

main()