import torch
import torch.nn as nn 
from model import get as get_model
import os
from argparse import ArgumentParser
def load_pretrained_model(model, model_path):
    """Load a pretrained model from a file.

    Args:
        model (torch.nn.Module): The model to load the weights into.
        model_path (str): The path to the file containing the weights.

    Returns:
        torch.nn.Module: The model with the weights loaded.
    """
    model.load_state_dict(torch.load(model_path))
    return model
def load_model_from_ckpt():
    # arg_model = "net_epcl"
    # cpkt_path = "/home/fangj1/Code/Vision-Language-on-3D-Scene-Understanding/EPCL/indoor_segmentation/checkpoints/epoch=062--mIoU_val=0.6972--.ckpt"
    # model = get_model(arg_model).load_from_checkpoint(checkpoint_path = cpkt_path)

    # # model.eval()
    # # model.freeze()
    # print(model.state_dict().keys())
    import torch

    # 指定 .ckpt 文件的路径
    checkpoint_path = '/home/fangj1/Code/Vision-Language-on-3D-Scene-Understanding/EPCL/indoor_segmentation/checkpoints/epoch=062--mIoU_val=0.6972--.ckpt'

    # 加载 .ckpt 文件
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))


    # 或者如果你想查看更详细的参数信息（如参数的形状和类型）
    if 'state_dict' in checkpoint:
        with open('parameter_info.txt', 'w') as f:
            for key, value in checkpoint['state_dict'].items():
                f.write(f"{key}: {value.size()}\n")  # 将每个参数的名字和尺寸写入文件
            
    # 检查是否有超参数
    if 'hyper_parameters' in checkpoint:
        with open('hyper_parameters.txt', 'w') as f:
            hyper_params = checkpoint['hyper_parameters']
            f.write(str(hyper_params) + '\n')  # 将超参数写入文件

    # 检查优化器状态
    if 'optimizer_states' in checkpoint:
        with open('optimizer_states.txt', 'w') as f:
            optimizer_states = checkpoint['optimizer_states']
            f.write(str(optimizer_states) + '\n')  # 将优化器状态写入文件



if __name__ == '__main__':
    load_model_from_ckpt()