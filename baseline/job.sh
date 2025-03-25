#!/bin/bash

#SBATCH --job-name=NLPgroup        # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=1        # Schedule one core
#SBATCH --time=01:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=scavenge    # Run on scavenge queue (all nodes, low priority)

# Print out the hostname of the node the job is running on
module load python/3.11
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/myenv/bin/activate


# Run your Python script
python baseline.py