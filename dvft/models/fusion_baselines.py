import torch
import torch.nn as nn
from .backbone import VisionBackbone


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.proj(x)


class ConcatFusion(nn.Module):
    def __init__(self, dim_b, dim_l, out_dim=768):
        super().__init__()
        self.proj_b = Projection(dim_b, out_dim)
        self.proj_l = Projection(dim_l, out_dim)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x_b, x_l):
        x_b = self.proj_b(x_b)
        x_l = self.proj_l(x_l)
        return self.head(torch.cat([x_b, x_l], dim=-1))


class AddFusion(nn.Module):
    def __init__(self, dim_b, dim_l, out_dim=768):
        super().__init__()
        self.proj_b = Projection(dim_b, out_dim)
        self.proj_l = Projection(dim_l, out_dim)
        self.head = nn.Linear(out_dim, out_dim)
    def forward(self, x_b, x_l):
        return self.head(self.proj_b(x_b) + self.proj_l(x_l))


class GatingFusion(nn.Module):
    def __init__(self, dim_b, dim_l, out_dim=768):
        super().__init__()
        self.proj_b = Projection(dim_b, out_dim)
        self.proj_l = Projection(dim_l, out_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.head = nn.Linear(out_dim, out_dim)
    def forward(self, x_b, x_l):
        fused = self.alpha * self.proj_b(x_b) + (1 - self.alpha) * self.proj_l(x_l)
        return self.head(fused)


class MLPFusion(nn.Module):
    def __init__(self, dim_b, dim_l, out_dim=768, hidden_dim=1024):
        super().__init__()
        self.proj_b = Projection(dim_b, out_dim)
        self.proj_l = Projection(dim_l, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x_b, x_l):
        x_b = self.proj_b(x_b)
        x_l = self.proj_l(x_l)
        return self.mlp(torch.cat([x_b, x_l], dim=-1))


class TransformerFusion(nn.Module):
    """双骨干拼接 + 4层Transformer"""
    def __init__(self, dim_b, dim_l, out_dim=768, num_layers=4, num_heads=8):
        super().__init__()
        self.proj_b = Projection(dim_b, out_dim)
        self.proj_l = Projection(dim_l, out_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=out_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(out_dim, out_dim)
    
    def forward(self, x_b, x_l):
        x_b = self.proj_b(x_b).unsqueeze(1)
        x_l = self.proj_l(x_l).unsqueeze(1)
        fused = torch.cat([x_b, x_l], dim=1)  # [B, 2, D]
        out = self.transformer(fused)
        pooled = out.mean(dim=1)
        return self.head(pooled)


class SingleBackboneAdapter(nn.Module):
    """单骨干 + 4层Transformer"""
    def __init__(self, backbone_name='vit_large_patch16_224', out_dim=768, 
                 num_layers=4, num_heads=8, num_classes=1000):
        super().__init__()
        self.backbone = VisionBackbone(backbone_name, pretrained=True, freeze_params=True)
        dim = self.backbone.output_dim
        self.proj = Projection(dim, out_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            dim_feedforward=out_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(out_dim, num_classes)
    
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat.mean(dim=1))
        feat = feat.unsqueeze(1)
        out = self.transformer(feat)
        return self.classifier(out.squeeze(1))