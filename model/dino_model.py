
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib


def patch_dinov2_input_channels(backbone, input_channels: int):
    """
    Replace DINOv2's patch embedding conv to accept N input channels.
    Initializes new channels by averaging the pretrained RGB weights.
    
    This is cleaner than an input projection layer.
    """
    old_proj = backbone.patch_embed.proj  # Conv2d(3, embed_dim, 14, 14)
    old_weight = old_proj.weight.data     # (embed_dim, 3, 14, 14)
    
    # New conv: same everything but input_channels instead of 3
    new_proj = nn.Conv2d(
        input_channels,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None
    )
    
    # Smart weight initialization:
    # Average existing RGB weights, tile across new channels
    avg_weight = old_weight.mean(dim=1, keepdim=True)  # (embed_dim, 1, 14, 14)
    new_weight = avg_weight.repeat(1, input_channels, 1, 1)  # (embed_dim, C, 14, 14)
    
    # Add small noise to break symmetry
    new_weight = new_weight + 0.02 * torch.randn_like(new_weight)
    
    new_proj.weight.data = new_weight
    if old_proj.bias is not None:
        new_proj.bias.data = old_proj.bias.data.clone()
    
    backbone.patch_embed.proj = new_proj
    print(f"  Replaced patch_embed: 3 → {input_channels} input channels")
    return backbone


class DINOv2SWEModelV2(nn.Module):
    """
    Cleaner version: modify patch embed directly, no input projection.
    """
    def __init__(
        self,
        input_channels: int = 8,
        dino_model: str = "dinov2_vitb14",
        img_size: int = 252,
        decoder_channels: int = 256,
        n_freeze_blocks: int = 6,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_grid_size = img_size // 14
        
        # Load backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', dino_model, pretrained=True
        )
        self.embed_dim = self.backbone.embed_dim
        
        # Modify patch embed for N-channel input
        self.backbone = patch_dinov2_input_channels(self.backbone, input_channels)
        
        # Freeze early blocks
        for i, block in enumerate(self.backbone.blocks):
            if i < n_freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        
        # Decoder (same as V1)
        self.decoder = self._build_decoder(decoder_channels)
    
    def _build_decoder(self, ch):
        g = self.patch_grid_size  # 18 for 252x252
        return nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, ch, 2, stride=2),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, ch // 2, 2, stride=2),
            nn.BatchNorm2d(ch // 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch // 2, ch // 4, 2, stride=2),
            nn.BatchNorm2d(ch // 4), nn.ReLU(inplace=True),
            nn.Upsample(size=(self.img_size, self.img_size), 
                        mode='bilinear', align_corners=False),
            nn.Conv2d(ch // 4, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        if H != self.img_size or W != self.img_size:
            x = F.interpolate(x, (self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        
        features = self.backbone.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']       # (B, N, D)
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
            B, self.embed_dim, self.patch_grid_size, self.patch_grid_size
        )
        
        out = self.decoder(patch_tokens)
        
        if H != self.img_size or W != self.img_size:
            out = F.interpolate(out, (H, W), mode='bilinear', align_corners=False)
        
        return out.squeeze(1)