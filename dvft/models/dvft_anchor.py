import torch
import torch.nn as nn
from .backbone import VisionBackbone


class CrossAttnCLS(nn.Module):
    """CLS token 作为 Query，关注 Patch tokens"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, cls_token, patches):
        B, D = cls_token.shape
        N = patches.size(1)
        
        q = self.q(cls_token).reshape(B, 1, self.num_heads, D//self.num_heads).transpose(1, 2)
        k = self.k(patches).reshape(B, N, self.num_heads, D//self.num_heads).transpose(1, 2)
        v = self.v(patches).reshape(B, N, self.num_heads, D//self.num_heads).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)
        return self.proj(out)


class CrossViTLayer(nn.Module):
    """双向交叉注意力层"""
    def __init__(self, dim, num_heads_s=4, num_heads_l=8):
        super().__init__()
        self.cross_s_to_l = CrossAttnCLS(dim, num_heads_s)
        self.cross_l_to_s = CrossAttnCLS(dim, num_heads_l)
        self.ln_s = nn.LayerNorm(dim)
        self.ln_l = nn.LayerNorm(dim)
    
    def forward(self, Zs, Zl):
        cls_s, ps_s = Zs[:, 0, :], Zs[:, 1:, :]
        cls_l, ps_l = Zl[:, 0, :], Zl[:, 1:, :]
        
        cls_s_new = cls_s + self.cross_s_to_l(self.ln_s(cls_s), self.ln_s(ps_l))
        cls_l_new = cls_l + self.cross_l_to_s(self.ln_l(cls_l), self.ln_l(ps_s))
        
        Zs_new = torch.cat([cls_s_new.unsqueeze(1), ps_s], dim=1)
        Zl_new = torch.cat([cls_l_new.unsqueeze(1), ps_l], dim=1)
        return Zs_new, Zl_new


class DVFTAnchor(nn.Module):
    """DVFT-Anchor: 基于CLS的渐进式双向校准"""
    def __init__(self, 
                 base_name='vit_base_patch16_224',
                 large_name='vit_large_patch16_224',
                 depth=4,
                 fusion_dim=768,
                 num_classes=1000):
        super().__init__()
        
        self.backbone_b = VisionBackbone(base_name, pretrained=True, freeze_params=True)
        self.backbone_l = VisionBackbone(large_name, pretrained=True, freeze_params=True)
        
        dim_b = self.backbone_b.output_dim
        dim_l = self.backbone_l.output_dim
        self.fusion_dim = fusion_dim
        
        self.proj_b = nn.Linear(dim_b, fusion_dim)
        self.proj_l = nn.Linear(dim_l, fusion_dim)
        
        self.layers = nn.ModuleList([
            CrossViTLayer(fusion_dim) for _ in range(depth)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim * 2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, x):
        Zb = self.backbone_b(x)
        Zl = self.backbone_l(x)
        
        Zb = self.proj_b(Zb)
        Zl = self.proj_l(Zl)
        
        for layer in self.layers:
            Zb, Zl = layer(Zb, Zl)
        
        cls_b = Zb[:, 0, :]
        cls_l = Zl[:, 0, :]
        fused = torch.cat([cls_b, cls_l], dim=-1)
        
        return self.classifier(fused)