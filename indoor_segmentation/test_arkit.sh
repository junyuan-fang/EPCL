#!/bin/bash
#SBATCH --job-name=epcl
#SBATCH --account=project_2002051
#SBATCH --partition=gpusmall
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
##SBATCH --gres=gpu:a100:1,nvme:900
#
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

python3 /scratch/project_2002051/junyuan/cvpr24-challenge/epcl/indoor_segmentation/test_arkit.py