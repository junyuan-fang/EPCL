from __future__ import print_function
import random
import shutil
import os
import glob
from copy import deepcopy
import open3d as o3d
import torch # why is it located here?
import numpy as np
from plyfile import PlyData
import pdb
import cv2
cv2.setNumThreads(0)
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import yaml
from easydict import EasyDict
from model import get as get_model
from model.utils.configs import Config
from model.utils.common_util import AverageMeter, intersectionAndUnion, find_free_port
import torch_scatter
# from model import get as get_model
# from dataset import get as get_dataset
# cuda_available = torch.cuda.is_available()
# print(f"CUDA Available: {cuda_available}")

# # If CUDA is available, print CUDA and cuDNN details
# if cuda_available:
#     print(f"PyTorch is using CUDA Version: {torch.version.cuda}")
#     print(f"cuDNN Version: {torch.backends.cudnn.version()}")

#     # Check if torch_scatter is using CUDA
#     scatter_cuda = torch_scatter.scatter_add(torch.tensor([1.0, 2.0], device='cuda'), 
#                                              torch.tensor([0, 0], device='cuda'))
#     print(f"torch_scatter CUDA check (should not fail): {scatter_cuda}")

# else:
#     print("CUDA is not available. Check your PyTorch installation and GPU.")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed=0
pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU

def read_txt(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [int(x.strip()) for x in lines]
    return lines

#parser = my_args()
args = Config()

    # ------------
    # randomness or seed
    # ------------
torch.backends.cudnn.benchmark = args.cudnn_benchmark

# load check point for the model
from importlib import import_module
args.load_model ="/scratch/project_2002051/junyuan/cvpr24-challenge/epcl/indoor_segmentation/checkpoints/epoch=062--mIoU_val=0.6972--.ckpt"
args.on_train = False
print('ckpt best. args.load_model=[{}]'.format(args.load_model))
assert args.load_model is not None, 'why did you come?'
print(args.transdown)
model = get_model(args.model).load_from_checkpoint(
    os.path.join(args.MYCHECKPOINT, args.load_model), 
    args=args).to(device) # args.strict_load

model.eval()
model.freeze()

###### process data

#data loader
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
class ArkitDataLoader(Dataset):
    def __init__(self, scene_path , query_path = None, mask_path = None):
        """
        初始化点云数据集
        :param ply_files: PLY文件的列表
        """
        self.scene_path = scene_path
        self.query_path = query_path
        self.mask_path = mask_path
        self.ply_list = []  # 先初始化列表
        self.query_list = []
        self.mask_list = []
        self.scene_id = None
        
        # 然后加载数据
        self.load_ply_list()
        self.load_query_list()
        self.load_mask_list()
        
    def load_ply_list(self):
        """
        加载PLY文件列表
        """
        # 读取PLY文件列表
        self.ply_list = sorted(glob.glob(os.path.join(self.scene_path, '4*/*.ply')))
        #print("Loaded PLY files:", self.ply_list)
    def load_query_list(self):
        if self.query_path is not None:
            # 加载CSV文件
            df = pd.read_csv(self.query_path)

            # 按照某个列的值进行排序
            df_sorted = df.sort_values(by='video_id')

            # 获取排序后的另一个列的值
            self.query_list = df_sorted['query'].values
            # # 输出获取的值
            # print(self.query_list)
            
    def load_mask_list(self):
        if self.mask_path is not None:
            self.mask_list = sorted(glob.glob(os.path.join(self.mask_path, '*.txt')))
            #print(self.mask_list)
    def __len__(self):
        """
        数据集中的样本数
        """
        return len(self.ply_list)

    def __getitem__(self, idx):
        """
        读取单个点云文件，并返回其数据
        :param idx: 索引
        """
        self.scene_id = self.mask_list[idx].split('/')[-1].split('_')[0]
        #print(self.ply_list[idx].split('/')[-1].split('_')[0])
        # 加载点云文件
        pcd = o3d.io.read_point_cloud(self.ply_list[idx])

        # 获取坐标
        coordinates = np.asarray(pcd.points, dtype=np.float32)

        # 获取特征，这里假设使用颜色作为特征
        if pcd.colors:
            features = np.asarray(pcd.colors, dtype=np.float32)  # RGB颜色
        else:
            features = np.zeros((coordinates.shape[0], 3), dtype=np.float32)  # 如果没有颜色，使用零填充

        
        #mask
        if self.mask_path is not None:
            mask = read_txt(self.mask_list[idx])
        # 将数据转换为torch tensors
        mask = torch.tensor(mask).unsqueeze(1)
        coordinates = torch.from_numpy(coordinates)
        features = torch.from_numpy(features)

        #return {'coord': coordinates, 'feat': features, 'prompt': self.query_list[idx], 'target': mask}
        return coordinates, features, self.query_list[idx], mask
def TrainValCollateFn(batch):
    coord, feat, prompt, mask = list(zip(*batch))
    offset, count = [], 0
    for item in coord: # len of pc
        count += item.shape[0]
        offset.append(count)
    
    print("Coordinates type:", type(coord[0]))  # Check the type of the first coordinate set
    print("Features type:", type(feat[0]))      # Check the type of the first features set
    print("Mask type:", type(mask[0]))          # Check the type of the first mask
    print("Offset type:", type(offset[0]))      # Check the type of the first offset
    
    data_dict = \
        {
            'coord': torch.cat(coord).to(device),
            'feat': torch.cat(feat).to(device),
            'target': torch.cat(mask).to(device),
            'prompt': list(prompt),
            'offset': torch.IntTensor(offset).to(device),
        }
    return data_dict
dataset = ArkitDataLoader(args.arkit_train_root,args.development_query_root,args.development_mask_root)
#data_loader = DataLoader(dataset, batch_size= args.train_batch, collate_fn=TrainValCollateFn)
data_loader = DataLoader(dataset, batch_size= args.train_batch, collate_fn=TrainValCollateFn)

data_iterator = iter(data_loader)
first_batch = next(data_iterator)
output = model(first_batch)#[{output: tensor()},{loss: tensor()}]

######################fine-tuning#####################


from torch import nn
cls = nn.Sequential(
    nn.Linear(32, 128), 
    nn.BatchNorm1d(128), 
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(128, 512), 
    nn.BatchNorm1d(512), 
    nn.ReLU(inplace=True))
cls = cls.to(device)

def print_cuda_memory_usage(device_id=0):
    t = torch.cuda.get_device_properties(device_id).total_memory
    r = torch.cuda.memory_reserved(device_id) 
    a = torch.cuda.memory_allocated(device_id)
    f = r - a  # free inside reserved

    print(f"CUDA Device ID: {device_id}")
    print(f"Total memory: {t / 1e9:.2f} GB")
    print(f"Reserved memory: {r / 1e9:.2f} GB")
    print(f"Allocated memory: {a / 1e9:.2f} GB")
    print(f"Free (inside reserved): {f / 1e9:.2f} GB")
    memory_summary = torch.cuda.memory_summary(device=device, abbreviated=False)
    print(memory_summary)
print_cuda_memory_usage(device)