#!/bin/bash
#SBATCH --job-name=epcl
#SBATCH --account=project_2002051
#SBATCH --partition=gputest
#SBATCH --time=00:12:00
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=10
##SBATCH --gres=gpu:a100_1g.5gb:1  # 请求一个分片的 A100 GPU
#SBATCH --gres=gpu:a100:1
## if local fast disk on a node is also needed, replace above line with:
##SBATCH --gres=gpu:a100:1,nvme:900
#
## Please remember to load the environment your application may need.
## And use the variable $LOCAL_SCRATCH in your batch job script 
## to access the local fast storage on each node.

start_time=$(date +"%Y-%m-%d %T")
echo "Started at: $start_time"
python3 test.py

python3 /scratch/project_2002051/junyuan/cvpr24-challenge/epcl/indoor_segmentation/test_arkit.py

current_time=$(date +"%Y-%m-%d %T")

# 检查上一个命令的退出状态
if [ $? -ne 0 ]; then
    current_time=$(date +"%Y-%m-%d %T")
    echo "Error occurred at: $current_time"  # 打印错误发生时间
else
    current_time=$(date +"%Y-%m-%d %T")
    echo "Finished successfully at: $current_time"
fi