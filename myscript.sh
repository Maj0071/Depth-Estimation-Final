#!/bin/bash

module load python/3.12
cat message.in > message.out
echo "Running Fourier script..." >> message.out

python3 run_fourier.py >> messge.out 2>&1
echo "Done." >> message.out

scontrol show job $SLURM_JOBID >> message.out