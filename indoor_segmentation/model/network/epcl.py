import math
import pdb
import random

import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter, scatter_softmax, scatter_sum, scatter_std, scatter_max
from easydict import EasyDict as edict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pointops2.functions import pointops
from .hier import *
from .inter import *
from .epcl_modules.transformer import EPCLEncoder, CLIPTextEncoder, CrossAttentionBlock

seed=0
# pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False 

def get_clip_text_feats(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    with torch.no_grad():
        texts_emb = clip.tokenize(texts).to(device)
        texts_feats = clip_model.encode_text(texts_emb)

    return texts_feats

# class CrossAttentionBlock(nn.Module):#num_heads = 8
#     def __init__(self, visual_feat_dim, text_feat_dim, output_dim):
#         super(CrossAttentionBlock, self).__init__()
#         self.visual_attention = nn.MultiheadAttention(embed_dim=visual_feat_dim, num_heads=8, batch_first=True)
#         self.text_attention = nn.MultiheadAttention(embed_dim=text_feat_dim, num_heads=8, batch_first=True)
#         self.fc = nn.Linear(visual_feat_dim + text_feat_dim, output_dim)

#     def forward(self, visual_feats, text_feats):
#         text_context, _ = self.visual_attention(query=text_feats, key=visual_feats, value=visual_feats)
#         visual_context, _ = self.text_attention(query=visual_feats, key=text_feats, value=text_feats)
#         combined_feats = torch.cat([visual_context + text_feats, visual_context + visual_feats], dim=-1)
#         output = self.fc(combined_feats)
#         return output

class BilinearFeedForward(nn.Module):

    def __init__(self, in_planes1, in_planes2, out_planes):
        super().__init__()
        self.bilinear = nn.Bilinear(in_planes1, in_planes2, out_planes)

    def forward(self, x):
        x = x.contiguous()
        x = self.bilinear(x, x)
        return x

####################################################################################

class NoIntraSetLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.out_planes = out_planes
        self.nsample = nsample

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]
        x_knn = x_knn[:, :, 3:]

        return (x, x_knn, knn_idx, p_r)

# PointMixerIntraSetLayer_ECCV22
class PointMixerIntraSetLayer(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3, 3, bias=False),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(3),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> n k b', 'sum', b=nsample))

        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes, mid_planes//share_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes//share_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes//share_planes, out_planes//share_planes, kernel_size=1),
            Rearrange('n c k -> n k c'))

        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3] # xyz->[n,k,3]

        energy = self.channelMixMLPs01(x_knn) # (n, k, k)
        
        p_embed = self.linear_p(p_r) # (n, k, out_planes)
        p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)

        energy = torch.cat([energy, p_embed_shrink], dim=-1) # (n, k, 2*k)
        energy = self.channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)

        x_v = self.channelMixMLPs03(x)  # (n, in_planes) -> (n, k)
        n = knn_idx.shape[0]; knn_idx_flatten = knn_idx.flatten()
        x_v  = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)

        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(
            n, nsample, self.share_planes, out_planes//self.share_planes)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, nsample, out_planes)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)

####################################################################################

class PointMixerBlock(nn.Module):
    expansion = 1

    def __init__(self, 
                 in_planes, planes, share_planes=8, 
                 nsample=16, 
                 use_xyz=False,
                 intraLayer='PointMixerIntraSetLayer',
                 interLayer='PointMixerInterSetLayer'):
        super().__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = nn.Sequential(
            globals()[intraLayer](planes, planes, share_planes, nsample),
            globals()[interLayer](in_planes, share_planes, nsample))
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes*self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]

class EPCLSegNet(nn.Module):
    mixerblock = PointMixerBlock

    def __init__(
        self, block, blocks, 
        c=6, k=13, nsample=[8,16,16,16,16], stride=[1,4,4,4,4],
        intraLayer='PointMixerIntraSetLayer',
        interLayer='PointMixerInterSetLayer',
        transup='SymmetricTransitionUpBlock', 
        transdown='TransitionDownBlock'):
        super().__init__()
        self.c = c
        self.intraLayer = intraLayer
        self.interLayer = interLayer
        self.transup = transup
        self.transdown = transdown
        self.in_planes, planes = c, [32, 64, 128, 256, 512]
        self.cross_blocks = 8# eight blocks
        self.cross_attention_block  =  CrossAttentionBlock(n_features=512, n_heads=8, n_hidden=512, dropout=0.1)

        # nn.ModuleList([
        #     CrossAttentionBlock(n_features=512, n_heads=8, n_hidden=512, dropout=0.1)
        #     for _ in range(self.cross_blocks)
        # ])
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        
        assert stride[0] == 1, 'or you will meet errors.'

        self.enc1 = self._make_enc(planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])  # N/4
        self.enc3 = self._make_enc(planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        
        # self.dec55 = self._make_dec(planes[0], 2, share_planes, nsample=nsample[0], is_head=True)  # transform p5
        # self.dec44 = self._make_dec(planes[1], 2, share_planes, nsample=nsample[1])  # fusion p5 and p4
        # self.dec33 = self._make_dec(planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        # self.dec22 = self._make_dec(planes[3], 2, share_planes, nsample=nsample[3])  # fusion p3 and p2
        # self.dec11 = self._make_dec(planes[4], 2, share_planes, nsample=nsample[4])  # fusion p2 and p1
        
        # self.cls = nn.Sequential(
        #     nn.Linear(planes[0], planes[0]), 
        #     nn.BatchNorm1d(planes[0]), 
        #     nn.ReLU(inplace=True), 
        #     nn.Linear(planes[0], k))
        self.mlp = nn.Sequential(
            nn.Linear(32, 64), 
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace=True), 
            nn.Linear(64, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace=True), 
            nn.Linear(128, 256), 
            nn.BatchNorm1d(256), 
            nn.ReLU(inplace=True), 
            nn.Linear(256, 512), 
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True), 
            nn.Linear(512, 512))
        
        
        # epcl encoder
        self.transformer = EPCLEncoder(embed_dim=768)
        self.text_transformer = CLIPTextEncoder(embed_dim=768)###
        self.input_layer =  nn.Sequential( 
                nn.Linear(planes[4], 768),
                nn.BatchNorm1d(768),
                nn.ReLU(inplace=True))
        self.output_layer =  nn.Sequential( 
                nn.Linear(768, planes[4]),
                nn.BatchNorm1d(planes[4]),
                nn.ReLU(inplace=True))
        self.build_loss_func()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100
    
    def _make_enc(self, planes, blocks, share_planes, stride, nsample):
        layers = []
        if planes == 512:
            layers.append(globals()['FixedSymmetricTransitionDownBlock']( 
                in_planes=self.in_planes, 
                out_planes=planes, 
                stride=stride, 
                nsample=nsample))
        else:
            layers.append(globals()[self.transdown]( 
                in_planes=self.in_planes, 
                out_planes=planes, 
                stride=stride, 
                nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                intraLayer=self.intraLayer,
                interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def _make_dec(self, planes, blocks, share_planes, nsample, is_head=False):
        layers = []
        layers.append(globals()[self.transup](
            in_planes=self.in_planes, 
            out_planes=None if is_head else planes, 
            nsample=nsample))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(self.mixerblock(
                in_planes=self.in_planes, 
                planes=self.in_planes, 
                share_planes=share_planes,
                nsample=nsample,
                intraLayer=self.intraLayer,
                interLayer=self.interLayer))
        return nn.Sequential(*layers)

    def pc_norm(self, pc):
        """ pc: [batch_size, num_points, num_channels], return [batch_size, num_points, num_channels] """
        centroid = torch.mean(pc, dim=1, keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc**2, dim=2, keepdim=True)), dim=1, keepdim=True).values
        pc = pc / m
        return pc
    
    def create_padding_mask(self,seq): #(batch_size, seq_len) output
        mask = (seq.sum(dim=-1) == 0)
        return mask

    def forward(self, pxo):#pxo -> point feature offset prompt = []
        p0, x0, o0, prompt = pxo  # (n, 3), (n, c), (b), (b)->xyz,feature,batch_size, barch_size这里的c和EPCLSegNet init 的 c=6 的c 不一样，这里的c 是 数据集的RGB， c=6 是xyzRGB
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)# x 这时候其实已经包括了自身x和p的信息。
        p1, x1, o1 = self.enc1([p0, x0, o0]) # 1 6 -> 32
        p2, x2, o2 = self.enc2([p1, x1, o1]) # 4 32 -> 64
        p3, x3, o3 = self.enc3([p2, x2, o2]) # 16 64 -> 128
        p4, x4, o4 = self.enc4([p3, x3, o3]) # 64 128 -> 256
        p5, x5, o5 = self.enc5([p4, x4, o4]) # 256 256 -> 512
        #print("for x shape",x1.shape, x2.shape, x3.shape,x4.shape, x5.shape)
        #for x torch.Size([341815, 32]) torch.Size([85453, 64]) torch.Size([21362, 128]) torch.Size([5340, 256]) torch.Size([1170, 512])
        #print("for p shape",p1.shape, p2.shape, p3.shape,p4.shape, p5.shape)

        #print("for o shape ",o1.shape, o2.shape, o3.shape,o4.shape, o5.shape)
        # for o torch.Size([2]) torch.Size([2]) torch.Size([2]) torch.Size([2]) torch.Size([2])
        #print("for o",o1, o2, o3,o4, o5)



        # ++++++++++++epcl module++++++++++++++++
        res = x5 #(n,512)
        b, n = len(o5), p5.shape[0]#b = batch_size, n = is all point, sum from all batches
        x5 = self.input_layer(x5)#project to 768  (n,768)
        pos = p5.reshape(b, n//b, -1)# from (n,3) to (b, n//b ,3) for latter positional encoding.
        pos = self.pc_norm(pos)# shift to origin
        _, x5, _ = self.transformer(x5.reshape(b, n//b, -1).permute(1,0,2), pos)#x5.reshape(b, n//b, -1).permute(1,0,2)->(n//b, b, 768) fit to ViT's transformer block
        #N,b,768
        x5 = x5.permute(1, 0, 2).reshape(n, -1)# b n//b 768 -> n,768
        x5 = self.output_layer(x5)# from 768 to 512
        # ///////////adding  ////////////////////////////////////////////////////////////////////////
        fine_txt_feature = self.text_transformer(text = prompt, apply_projection = False )
        #print("fine_txt_feature:", fine_txt_feature.shape)# torch.Size([b,77,512])
        source_padding_mask = self.create_padding_mask(fine_txt_feature)
        #print("source_padding_mask", source_padding_mask.shape)#([2, 7])
        global_txt_feature = self.text_transformer(text = prompt, apply_projection = True )
        #print("global_txt_featrue",global_txt_feature.shape)# torch.Size([b, 512])
        # ///////////////////////////////////////
        # /////////// fine tuning  ////////////////
        b_attension = []
        previous_offset_idx = 0
        for idx, offset_idx in enumerate(o5):
            #print(x5[previous_offset_idx:offset_idx, :].shape)
            x5_ = self.cross_attention_block(
                query=x5[previous_offset_idx:offset_idx, :], # torch.Size([n, 512])
                kv=fine_txt_feature[idx,:,:],# torch.Size([b, 512])
                src_mask = source_padding_mask[idx,:]
            )
            b_attension.append(x5_)
            previous_offset_idx = offset_idx
            
        b_attension = torch.cat(b_attension, dim=0)  # 将所有张量拼接成一个大张量        
        # //////////////////////////////////////////////////////////////////////////////////////////
        x5 = res + b_attension
        # +++++++++++++++++++++++++++++++++++++++
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1] # 256 512
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1] # 64 512->256
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1] # 16 256->128
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1] # 4 128->64 
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1] # 1 64-32
        # /////////// add and fine tuning  ////////////////
        # x2 = self.dec21[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1] # 1 64->32
        # /////////// fine tuning  ////////////////
        #print("shape of x1", x1.shape)#torch.Size([341815, 32])
        # n 32 from x1+ n 512 form x5_
        # mlp input is (n, 32 from conv +512 from cross attension ), output is (n, 512)
        #x = self.cls(x1) # [n,13]
        x = self.mlp(x1)
        global_txt_feature /= global_txt_feature.norm(dim=-1, keepdim=True)#torch.Size([b, 512])
        x /= x.norm(dim=-1, keepdim=True)#torch.Size([341815, 512])
        
        logit_scale = self.logit_scale.exp()
        #go throght txt_featrue batch 
        b_output = []
        previous_offset_idx = 0
        for idx, offset_idx in enumerate(o0):
            x5_ = x[previous_offset_idx:offset_idx,:]
            current_global_txt_feature = global_txt_feature[idx,:]##torch.Size([1, 512])
            similarity = (logit_scale* x5_ @ current_global_txt_feature.T)##torch.Size([slised341815, 1])
            previous_offset_idx = offset_idx
            print(similarity.shape)
            b_output.append(similarity)
        output = torch.cat(b_output, dim=0)  # 将所有张量拼接成一个大张量
        print(output.dtype)
        output = output.unsqueeze(1)
    
            
        #similarity = (global_txt_feature @ x.T).softmax(dim=-1)
        # threshold = 0.25
        # binary_mask = similarity > threshold
        return output#similarity 

def getEPCLSegNet(**kwargs):
    model = EPCLSegNet(PointMixerBlock, [2, 3, 4, 6, 3], **kwargs)  
    
    return model
