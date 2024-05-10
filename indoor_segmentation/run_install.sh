#!/bin/bash
#SBATCH --job-name=epcl
#SBATCH --account=project_2002051
#SBATCH --partition=gpusmall
#SBATCH --time=00:40:00
##SBATCH --ntasks=1
##SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
##SBATCH --gres=gpu:a100:1,nvme:900
#
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

# python3 /scratch/project_2002051/junyuan/cvpr24-challenge/epcl/indoor_segmentation/print_cuda.py
# cd /scratch/project_2002051/junyuan/cvpr24-challenge/epcl/indoor_segmentation/lib/pointops2/
# python3 /scratch/project_2002051/junyuan/cvpr24-challenge/epcl/indoor_segmentation/lib/pointops2/setup.py install
# pip3 install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install torch-scatter==2.0.9 --force-reinstall -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
