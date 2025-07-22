#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --job-name=FOM-calculation
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G

#SBATCH --error=errors/%x-%j.err
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-user=john.waiton@postgrad.manchester.ac.uk
#SBATCH --mail-type=ALL


RUNNUMBER='15153, 15152, 15151'
TIMESTAMP='210625, 220625, 230625'
FOM_TIMESTAMP='20251707'
CITY='thekla'



echo "Running FOM calculator 170725.."
echo "================"
echo "RUN NUMBER    : ${RUNNUMBER}"
echo "TIMESTAMP     : ${TIMESTAMP}"
echo "CITY          : ${CITY}"
echo "================"

python3 FOM_170725.py "${RUNNUMBER}" "${TIMESTAMP}" "${CITY}" "${FOM_TIMESTAMP}"


echo "DONE!!!"
