import clip
import torch

class ModifiedCLIP(torch.nn.Module):
    def __init__(self, original_model):
        super(ModifiedCLIP, self).__init__()
        self.token_embedding = original_model.token_embedding
        self.positional_embedding = original_model.positional_embedding
        self.transformer = original_model.transformer
        self.ln_final = original_model.ln_final
        self.dtype = next(original_model.parameters()).dtype  # 取得数据类型

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # 提取"end of text" token的特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return x

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
original_model, preprocess = clip.load('ViT-B/32', device=device)

# 创建修改后的模型实例
modified_clip = ModifiedCLIP(original_model)

# 创建文本输入
text = clip.tokenize(["a photo of a cat", "a photo of a dog"]).to(device)

# 获取编码器输出
with torch.no_grad():
    text_features = modified_clip.encode_text(text)

# 检查特征维度
print("Feature dimensions:", text_features.shape)
