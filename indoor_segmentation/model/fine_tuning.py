from net_epcl import net_epcl
import torch
import torch.nn as nn

class CrossAttentionBlock(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        
    def forward(self, visual_features, text_features):
        # Cross-attention: Query来自text features，Key和Value来自visual features
        attn_output, attn_output_weights = self.attn(text_features, visual_features, visual_features)
        return attn_output

class FineTuning(net_epcl):
    def __init__(self, args):
        super(FineTuning, self).__init__(args)
        
        # 假设visual和text特征维度相同，且我们使用8个head
        self.cross_att_blocks = nn.ModuleList([CrossAttentionBlock(feature_dim=512, num_heads=8) for _ in range(8)])
        
    def forward(self, data_dict):
        # 假设原始forward方法已经为我们生成了visual_features和text_features
        visual_features, text_features = super().forward(data_dict)
        
        # 应用cross-attention blocks
        for block in self.cross_att_blocks:
            text_features = block(visual_features, text_features)
        
        # 这里你可以继续处理text_features或将其返回
        return text_features



# def load_pretrained_weights(model, checkpoint_path):
#     # 加载预训练的权重
#     model.load_state_dict(torch.load(checkpoint_path), strict=False)
#     return model

# # 初始化你的模型和参数
# args = YourArgumentsSetupFunction()
# finetuning_model = FinetuningEPCL(args)

# # 载入预训练模型
# checkpoint_path = 'path_to_pretrained_model.pth'
# finetuning_model = load_pretrained_weights(finetuning_model, checkpoint_path)
