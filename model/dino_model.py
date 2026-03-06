
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def patch_dinov2_input_channels(backbone, input_channels: int):
    """
    Replace DINOv2's patch embedding conv to accept N input channels.
    Initializes new channels by averaging the pretrained RGB weights.
    
    This is cleaner than an input projection layer.
    """
    # Point to cache explicitly — no internet needed
    os.environ['TORCH_HOME'] = '/home/cmbreen/.cache/torch'
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
    

# pip install transformers
# or load from HuggingFace: ibm-nasa-geospatial/Prithvi-100M

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class PrithviSWEModel(nn.Module):
    """
    Prithvi-100M encoder + CNN decoder for dense SWE regression.
    
    Key differences from DINOv2 version:
    - No input projection needed (we map our channels to Prithvi's 6)
    - No resize needed (256x256 works natively, 256/16=16 patches)  
    - Temporal dimension handled explicitly
    """
    
    def __init__(
        self,
        input_channels: int = 8,        # your 8 channels
        prithvi_channels: int = 6,       # Prithvi expects 6 bands
        img_size: int = 256,             # your patch size, works natively
        decoder_channels: int = 256,
        n_freeze_blocks: int = 6,
        num_frames: int = 1,             # T=1 for single-date
    ):
        super().__init__()
        
        self.img_size = img_size
        self.num_frames = num_frames
        self.patch_size = 16
        self.patch_grid = img_size // self.patch_size  # 16
        
        # --------------------------------------------------------
        # 1) Channel projection: your 8 channels → Prithvi's 6
        #    This is MUCH more natural than DINOv2's 8→3 projection
        #    because Prithvi's 6 bands have real physical meaning
        # --------------------------------------------------------
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, prithvi_channels, kernel_size=1),
        )
        
        # --------------------------------------------------------
        # 2) Load Prithvi backbone
        # --------------------------------------------------------
        print("Loading Prithvi-100M from HuggingFace...")
        self.backbone = AutoModel.from_pretrained(
            "ibm-nasa-geospatial/Prithvi-100M",
            trust_remote_code=True,
        )
        self.embed_dim = 768  # Prithvi-100M embed dim
        
        # --------------------------------------------------------
        # 3) Freeze early blocks
        # --------------------------------------------------------
        for i, block in enumerate(self.backbone.blocks):
            if i < n_freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False
        
        # --------------------------------------------------------
        # 4) Decoder — same structure as DINOv2 version
        #    patch_grid=16 for 256x256 input (vs 18 for DINOv2 252x252)
        # --------------------------------------------------------
        g = self.patch_grid  # 16
        self.decoder = nn.Sequential(
            # 16x16 → 32x32
            nn.ConvTranspose2d(self.embed_dim, decoder_channels, 2, stride=2),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(decoder_channels, decoder_channels // 2, 2, stride=2),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            # 64x64 → 128x128
            nn.ConvTranspose2d(decoder_channels // 2, decoder_channels // 4, 2, stride=2),
            nn.BatchNorm2d(decoder_channels // 4),
            nn.ReLU(inplace=True),
            # 128x128 → 256x256
            nn.ConvTranspose2d(decoder_channels // 4, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Final head
            nn.Conv2d(64, 1, kernel_size=1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.decoder.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — your input channels
        Returns:
            (B, H, W) — predicted SWE
        """
        B, C, H, W = x.shape
        
        # 1) Project to 6 channels
        x_6band = self.input_proj(x)  # (B, 6, H, W)
        
        # 2) Add temporal dimension: (B, 6, H, W) → (B, T, 6, H, W)
        x_temporal = x_6band.unsqueeze(1)  # (B, 1, 6, H, W)
        
        # 3) Prithvi forward — returns patch tokens
        outputs = self.backbone(x_temporal)
        
        # patch tokens: (B, T*num_patches, embed_dim)
        # For T=1, 16x16 grid: (B, 256, 768)
        patch_tokens = outputs.last_hidden_state  
        
        # Remove temporal dim from patches (T=1 so just reshape)
        # (B, 256, 768) → (B, 768, 16, 16)
        patch_tokens = patch_tokens[:, :self.patch_grid**2, :]  # drop CLS if present
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(
            B, self.embed_dim, self.patch_grid, self.patch_grid
        )
        
        # 4) Decode to full resolution
        out = self.decoder(patch_tokens)  # (B, 1, 256, 256)
        
        return out.squeeze(1)  # (B, H, W)