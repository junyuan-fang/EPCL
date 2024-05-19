
from copy import deepcopy
import pdb
import os
import random
import glob

import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from dataset.utils.data_util import data_prepare_v101 as data_prepare
from dataset.utils.data_util import sa_create
from dataset.utils import transform as t

seed=0
# pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
#data loader
import torch
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
        #(N,3),(N,3), [str], (N,1)
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
            'coord': torch.cat(coord),#.to(device),
            'feat': torch.cat(feat),#.to(device),
            'target': torch.cat(mask),#.to(device),
            'prompt': list(prompt),
            'offset': torch.IntTensor(offset)#.to(device),
        }
    return data_dict