import torch
import torch.nn as nn
import timm
import os
import safetensors.torch


class VisionBackbone(nn.Module):
    """统一的ViT骨干网络封装，输出三维序列 [B, L, C]"""
    
    def __init__(self, model_name, pretrained=True, freeze_params=True):
        super().__init__()
        self.model_name = model_name
        self.freeze_params = freeze_params
        
        # 创建模型
        model_kwargs = {'pretrained': pretrained, 'num_classes': 0}
        self.model = timm.create_model(model_name, **model_kwargs)
        
        # 加载本地权重（如果提供路径）
        if isinstance(pretrained, str) and os.path.isfile(pretrained):
            self._load_local_weights(pretrained)
        
        # 冻结参数
        if freeze_params:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # 获取输出维度
        self.output_dim = self._get_output_dim()
        self.is_vit = model_name.startswith(("vit_", "deit_", "dinov2_"))
    
    def _load_local_weights(self, path):
        if path.endswith('.safetensors'):
            state_dict = safetensors.torch.load_file(path)
            self.model.load_state_dict(state_dict, strict=False)
        elif path.endswith('.pt') or path.endswith('.pth'):
            state_dict = torch.load(path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict, strict=False)
    
    def _get_output_dim(self):
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat = self._extract_features(dummy)
            return feat.shape[-1]
    
    def _extract_features(self, x):
        if hasattr(self.model, 'forward_features'):
            feat = self.model.forward_features(x)
        else:
            feat = self.model(x)
        
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        
        if feat.dim() == 4:  # [B, C, H, W] -> [B, H*W, C]
            B, C, H, W = feat.shape
            feat = feat.permute(0, 2, 3, 1).reshape(B, H*W, C)
        elif feat.dim() == 2:
            feat = feat.unsqueeze(1)
        
        return feat
    
    def forward(self, x):
        feat = self._extract_features(x)
        return feat