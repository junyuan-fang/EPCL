import torch
from torch import nn
import clip

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
        emb_dim = kwargs.get("embed_dim")
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
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = 'ViT-B/32'
        clip_model = clip.load(clip_model_name, device=device)[0]

        clip_vit = clip_model.visual
        clip_vit.float()
        if freeze:
            print("------------------- frozen ---------------------")
            clip_vit.eval()
            for k, v in clip_vit.named_parameters():
                v.requires_grad = False
                print(k)
            print("------------------- frozen ---------------------")
        return clip_vit.transformer, clip_vit.class_embedding, clip_vit.positional_embedding[0]

    def get_prompt(self, batch_size, te_token, te_encoder, device):
        prompts_tokens = te_token.expand(batch_size,
                                        -1).view(batch_size, -1).to(device)
        past_key_values = te_encoder(
            prompts_tokens)

        return past_key_values

    def forward(self, feats, xyz):
       
        N, B, C = feats.shape
        feats = feats.permute(1, 0, 2)  # [B, N, C]
        pos = self.pos_embed(xyz)

        cls_tok = self.cls_token.expand(B, 1, C)
        cls_pos = self.cls_pos.expand(B, 1, C)
        
        feats = torch.cat([cls_tok, feats], dim=1)   # [B, N+1, C]
        pos = torch.cat([cls_pos, pos], dim=1)
        feats = feats + pos
        task_emb = self.get_prompt(batch_size=B,
                                   te_token=self.te_tok,
                                   te_encoder=self.te_encoder,
                                   device=feats.device)
        feats = torch.cat([feats, task_emb], dim=1)
        # load frozen clip transformer
        new_feats = self.blocks(feats)
    
        return xyz, new_feats[:, 1:N+1, :].permute(1, 0, 2), None#exclude cls token