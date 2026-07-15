import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from .backbone import VisionBackbone


class AlternatingAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_dim=2048, dropout=0.1, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, base_features, large_features):
        # 自注意力
        norm_q = self.self_norm(queries)
        attn_out, _ = self.self_attn(norm_q, norm_q, norm_q)
        queries = queries + self.dropout(attn_out)
        
        # 交叉注意力 (交替: 偶数层关注Base，奇数层关注Large)
        visual_features = base_features if self.layer_idx % 2 == 0 else large_features
        norm_q = self.cross_norm(queries)
        cross_out, _ = self.cross_attn(norm_q, visual_features, visual_features)
        queries = queries + self.dropout(cross_out)
        
        # FFN
        norm_q = self.ffn_norm(queries)
        ffn_out = self.ffn(norm_q)
        queries = queries + self.dropout(ffn_out)
        
        return queries


class GlobalFusionLayer(nn.Module):
    """最终全局融合层 (不带自注意力)"""
    def __init__(self, hidden_size, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_size),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, visual_concat):
        norm_q = self.cross_norm(queries)
        cross_out, _ = self.cross_attn(norm_q, visual_concat, visual_concat)
        queries = queries + self.dropout(cross_out)
        
        norm_q = self.ffn_norm(queries)
        ffn_out = self.ffn(norm_q)
        queries = queries + self.dropout(ffn_out)
        return queries


class DVFTQuery(nn.Module):
    """DVFT-Query: 基于可学习Query的交替提取"""
    def __init__(self,
                 base_name='vit_base_patch16_224',
                 large_name='vit_large_patch16_224',
                 hidden_size=768,
                 num_query_tokens=16,
                 num_heads=8,
                 num_alt_layers=4,
                 num_classes=1000):
        super().__init__()
        
        self.backbone_b = VisionBackbone(base_name, pretrained=True, freeze_params=True)
        self.backbone_l = VisionBackbone(large_name, pretrained=True, freeze_params=True)
        
        dim_b = self.backbone_b.output_dim
        dim_l = self.backbone_l.output_dim
        
        self.proj_b = nn.Linear(dim_b, hidden_size)
        self.proj_l = nn.Linear(dim_l, hidden_size)
        
        # 可学习Query
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
        trunc_normal_(self.query_tokens, std=0.02)
        
        # Type Embedding (区分Base和Large特征)
        self.type_embed = nn.Embedding(2, hidden_size)
        
        # 交替注意力层
        self.alt_layers = nn.ModuleList([
            AlternatingAttentionLayer(hidden_size, num_heads, layer_idx=i)
            for i in range(num_alt_layers)
        ])
        
        # 全局融合层
        self.global_layer = GlobalFusionLayer(hidden_size, num_heads)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )
    
    def _add_type_embed(self, features, type_id):
        L = features.size(1)
        type_ids = torch.full((L,), type_id, dtype=torch.long, device=features.device)
        type_emb = self.type_embed(type_ids)
        return features + type_emb.unsqueeze(0)
    
    def forward(self, x):
        B = x.size(0)
        
        # 提取特征
        Zb = self.backbone_b(x)
        Zl = self.backbone_l(x)
        
        # 去除CLS token + 投影 + Type Embedding
        base_feat = self.proj_b(Zb[:, 1:, :])
        large_feat = self.proj_l(Zl[:, 1:, :])
        base_feat = self._add_type_embed(base_feat, 0)
        large_feat = self._add_type_embed(large_feat, 1)
        
        # 初始化Queries
        queries = self.query_tokens.expand(B, -1, -1)
        visual_concat = torch.cat([base_feat, large_feat], dim=1)
        
        # 交替注意力
        for layer in self.alt_layers:
            queries = layer(queries, base_feat, large_feat)
        
        # 全局融合
        queries = self.global_layer(queries, visual_concat)
        
        # 池化 + 分类
        pooled = queries.mean(dim=1)
        return self.classifier(pooled)