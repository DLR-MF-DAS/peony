#!/bin/bash

#SBATCH -J upsampling
#SBATCH -o {{ workdir }}/%x.%j.%N.out
#SBATCH -D  {{ workdir }}/
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --get-user-env
#SBATCH --nodes=1-1
#SBATCH --cpus-per-task=1
#SBATCH --export=NONE
#SBATCH --time=08:00:00

module load slurm_setup
module load charliecloud
/opt/snap/bin/gpt /peony/test/bandselect_upsampling.xml -Pinput={{ path }} -Poutput={{ workdir }}/upsampled.tif
