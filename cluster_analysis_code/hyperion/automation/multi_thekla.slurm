#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=thekla
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=32G

#SBATCH --error=errors/%x-%j.err
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL

module load parallel

export blobR_values="0.1 0.2 0.3 0.4 0.5"
export scanR_values="1 2 3 4 5"

# Number of CPUs per job
CPUS_PER_JOB=4

# Export OMP_NUM_THREADS to ensure each job uses the allocated CPUs
export OMP_NUM_THREADS=$CPUS_PER_JOB

# Run parallel with --slots to allocate CPUs per job
parallel --jobs $((SLURM_CPUS_PER_TASK / CPUS_PER_JOB)) --slots $CPUS_PER_JOB ./run_thekla.sh --blobR {1} --scanR {2} ::: $blobR_values ::: $scanR_values