#!/bin/bash

#SBATCH -J upsampling
#SBATCH -o $1/%x.%j.%N.out
#SBATCH -D $1/
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --get-user-env
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --export=NONE
#SBATCH --time=08:00:00

module load slurm_setup
module load charliecloud
/opt/snap/bin/gpt /peony/test/bandselect_upsampling.xml -Pinput=$2 -Poutput=$1/upsampled.tif
