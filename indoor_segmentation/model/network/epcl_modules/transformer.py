import torch
from torch import nn
import clip
from torch.nn import MultiheadAttention, LayerNorm

class TaskEmbEncoder(torch.nn.Module):
    def __init__(
            self,
            token_num=40,
            emb_dim=384
    ):
        super().__init__()
        # Use a two-layer MLP to encode the prefix
        self.embedding = torch.nn.Embedding(token_num, emb_dim)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, te: torch.Tensor):
        te_tokens = self.embedding(te)
        past_key_values = self.trans(te_tokens)

        return past_key_values

class EPCLEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_dim = kwargs.get("embed_dim")#768
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, emb_dim),
        )
        num_tokens = 13
        self.te_tok = torch.arange(num_tokens).long()
        self.te_encoder = TaskEmbEncoder(
            token_num=num_tokens,
            emb_dim=emb_dim
            )
        self.blocks, self.cls_token, self.cls_pos = self.clip_transformer()
        


    def clip_transformer(self, freeze=True):
    
        clip_model_name = 'ViT-B/32'
        clip_model = clip.load(clip_model_name, device=self.device)[0]

        clip_vit = clip_model.visual
        clip_vit.float()
        if freeze:
            print("------------------- frozen ---------------------")
            print("clip_model.visual")
            clip_vit.eval()
            for k, v in clip_vit.named_parameters():
                v.requires_grad = False
                #print(k)
            print("------------------- frozen ---------------------")
            #transformer's input output are the same(sequence length, 1, feature dimension), 
            #class_embedding
        return clip_vit.transformer, clip_vit.class_embedding, clip_vit.positional_embedding[0]# ()

    def get_prompt(self, batch_size, te_token, te_encoder, device):
        prompts_tokens = te_token.expand(batch_size,
                                        -1).view(batch_size, -1).to(device)
        past_key_values = te_encoder(
            prompts_tokens)

        return past_key_values

    def forward(self, feats, xyz):
       
        N, B, C = feats.shape   #xyz.shape = [B,N,C]
        feats = feats.permute(1, 0, 2)  # [B, N, C] = # [B, N, 768]
        pos = self.pos_embed(xyz) # # [B, N, 3]

        #print("cls_token", self.cls_token.shape)#([768])
        cls_tok = self.cls_token.expand(B, 1, C)
        #print("cls_token",cls_tok.shape)
        #print("cls_pos_token", self.cls_pos.shape)#([768])
        cls_pos = self.cls_pos.expand(B, 1, C)
        #print("cls_pos_token", cls_pos.shape)#([2, 1, 768])

        
        feats = torch.cat([cls_tok, feats], dim=1)   # [B, 1+N, C]
        pos = torch.cat([cls_pos, pos], dim=1) #(B, 1+N, C)
        feats = feats + pos
        task_emb = self.get_prompt(batch_size=B,
                                   te_token=self.te_tok,
                                   te_encoder=self.te_encoder,
                                   device=feats.device)
        #print("feats", feats.shape)#([2, 5545, 768])
        feats = torch.cat([feats, task_emb], dim=1)##(B, 1+N+13, C)
        #print("feats", feats.shape)#([2, 5558, 768])
        # load frozen clip transformer
        new_feats = self.blocks(feats)
        #print("new_feats", new_feats.shape)#([2, 5558, 768])#*(B,N,C)
    
        return xyz, new_feats[:, 1:N+1, :].permute(1, 0, 2), None#exclude cls token (N,B,C)
    
    
class CLIPTextEncoder(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_dim = kwargs.get("embed_dim")#768
        self.token_embedding, self.positional_embedding, self.transformer, self.ln_final, self.text_projection = self.clip_txt_transformer()

    # def clip_txt_transformer(self, freeze=True):
    
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     clip_model_name = 'ViT-B/32'
    #     clip_model = clip.load(clip_model_name, device=device)[0]

    #     return clip_model.token_embedding, clip_model.positional_embedding, clip_model.transformer, clip_model.ln_final, clip_model.text_projection
    def clip_txt_transformer(self, freeze=True):
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = 'ViT-B/32'
        clip_model = clip.load(clip_model_name, device=device)[0].float()

        return clip_model.token_embedding, clip_model.positional_embedding, clip_model.transformer, clip_model.ln_final, clip_model.text_projection
    
    #@property
    def encode_text(self, text, apply_projection=True):
        text = clip.tokenize(text).to(self.device)
        x = self.token_embedding(text)  # Convert to float
        print("embedding shape",x.shape)
        x = x + self.positional_embedding  # Convert to float
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)  # Convert to float
        print("transformer output",x.shape) #torch.Size([77, 2, 512])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # Convert to float 
        print("after layer norm", x.shape) # torch.Size([2, 77, 512])
        if apply_projection:
            # 只在需要时应用投影层
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection #nn.Linear(768, 512) project 768 to 512 
            print("after projection", x.shape)
        else:
            # 返回没有应用投影层的特征
            # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] 
            # print("with out projection", x.shape)
            # 返回EOT及其之前所有的token的特征
            eot_indices = text.argmax(dim=-1)
            output_features = []
            for i, eot_index in enumerate(eot_indices):
                output_features.append(x[i, :eot_index + 1, :])
            # 用pad_sequence将所有序列对齐到相同长度
            output_features = torch.nn.utils.rnn.pad_sequence(output_features, batch_first=True) # 默认0填充
            print("without projection, all token features", output_features.shape)
            return output_features



        return x

    def forward(self, text, apply_projection=True):
        print(text)
        text_features = self.encode_text(text, apply_projection=apply_projection)
        return text_features
    


class CrossAttentionBlock(nn.Module):
    def __init__(self, n_features=512, n_heads=8, n_hidden=512, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.MHselfA = nn.MultiheadAttention(embed_dim=n_features, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.preLN1 = nn.LayerNorm(n_features)
        
        self.MHcrossA = nn.MultiheadAttention(embed_dim=n_features, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.preLN2 = nn.LayerNorm(n_features)
        self.FF = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=n_hidden, out_features=n_features)
        )
        self.postLN = nn.LayerNorm(n_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, kv, src_mask):
        tgt = query
        memory = kv
        
        # Self Attention
        tgt2 = self.MHselfA(tgt, tgt, tgt, key_padding_mask=None)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.preLN1(tgt)

        # Cross Attention
        tgt2 = self.MHcrossA(tgt, memory, memory, key_padding_mask=src_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.preLN2(tgt)
        
        # Feedforward
        tgt2 = self.FF(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.postLN(tgt)
        
        return tgt